"""
plot_horizon_error.py  –  Plots/

Compares open-loop force-prediction error vs. rollout horizon step
for models trained with different BPTT horizons (k=5, 10, 25).

For each model:
  1. Encode every window in the eval dataset  →  z_0
  2. Roll out the dynamics model for H steps with the ground-truth
     control sequence  (no new observations)
  3. Decode each z_h  →  [Cd, Cl]  and compute MAE vs. ground truth
  4. Average MAE over all sequences  →  one curve per model

Usage:
    python Plots/plot_horizon_error.py

Set the three MODEL_IDS below once you have the timestamps from training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from libs.models import TemporalEncoder, LatentDynamicsModel, ForceDecoder
from libs.data import get_prepared_data, loadData
from libs.test_encoder_predictor import (
    find_model_paths, find_checkpoint_files,
    load_checkpoint,
)
from parameters import Args

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ---------------------------------------------------------------------------
# USER SETTINGS  ← fill in the timestamps once training is done
# ---------------------------------------------------------------------------

MODEL_IDS = {
    r'$k=3$': '20260328_17_22_00',
    r'$k=5$':  '20250906_19_38_58',
    r'$k=10$': '20260328_15_48_15',
    r'$k=25$': '20260328_15_52_32',
}

HORIZON         = 50       # max rollout steps to evaluate
CHECKPOINTS_DIR = '../03_Checkpoints'
OUTPUT_DIR      = Path('../04_Results/paper_figures/')
OUTPUT_NAME     = 'horizon_error_comparison'

# Eval dataset (same as in parameters.py)
EVAL_DATASET    = ['../01_Data/jet_2Dtruck_01092024_chirp15u_9999.hdf5']

# Colors consistent with the rest of the paper
COLORS = ['tab:green', 'k', 'tab:orange', 'tab:red']

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_model(model_id, checkpoints_dir, device):
    """Load encoder, dynamics and force decoder for a given model ID."""
    case_ckp_dir, _ = find_model_paths(model_id, checkpoints_dir)
    if case_ckp_dir is None:
        raise FileNotFoundError(f"Checkpoints not found for model {model_id}")

    enc_ckpt, dyn_ckpt, force_ckpt = find_checkpoint_files(case_ckp_dir, model_id)

    # Load Args from the checkpoint to reconstruct the exact architecture
    import json
    ckpt_data = torch.load(enc_ckpt, map_location=device, weights_only=True)
    if 'args' in ckpt_data:
        args = Args(**json.loads(ckpt_data['args']))
    else:
        args = Args()

    # Infer input_dim from LSTM weight: shape is (4*hidden_dim, input_dim)
    input_dim = ckpt_data['state_dict']['lstm.weight_ih_l0'].shape[1]

    encoder = TemporalEncoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.enc_hidden_dim,
        num_layers=args.n_layers,
        dropout_rate=args.dropout,
    ).to(device)

    dynamics = LatentDynamicsModel(
        latent_dim=args.latent_dim,
        action_dim=1,
        hidden_dim=args.dyn_hidden_dim,
        use_residual=args.residual_predictor,
    ).to(device)

    force_dec = ForceDecoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.forces_hidden_dim,
        out_dim=2,
        dropout_rate=args.forces_dropout,
        arch=args.force_decoder_arch,
    ).to(device)

    load_checkpoint(encoder,   enc_ckpt,   device, verbose=False)
    load_checkpoint(dynamics,  dyn_ckpt,   device, verbose=False)
    load_checkpoint(force_dec, force_ckpt, device, verbose=False)

    encoder.eval()
    dynamics.eval()
    force_dec.eval()

    return encoder, dynamics, force_dec, args


def compute_horizon_error(encoder, dynamics, force_dec, args,
                          eval_dataset, horizon, device):
    """
    Returns mae_cd, mae_cl, nmae_cd, nmae_cl  -  arrays of shape (horizon,),
    step 1..horizon. mae_* are the plain L1 (MAE) per-step errors; nmae_* are
    the L1/sigma errors, i.e. the per-step MAE divided by the standard deviation
    of the corresponding reference force coefficient over the whole eval set,
    matching the L1/sigma definition used in sec:modelPerf of the manuscript.
    """
    # Build a copy of args pointing at the eval dataset, n_test=0
    import copy
    eval_args = copy.copy(args)
    eval_args.datafile      = eval_dataset
    eval_args.n_test        = 0
    eval_args.augment_with_symmetry = False   # no augmentation for eval

    dataloader_eval, _ = get_prepared_data(eval_args, device,
                                           shuffle_train=False, shuffle_test=False)

    # We need forces scaling — re-load via loadData
    from libs.data import loadData
    (_, forces_unscaled, _, _, _, forces_mean, forces_std,
     _, _, _, _) = loadData(eval_args)

    forces_mean_t = torch.tensor(np.array(forces_mean), device=device, dtype=torch.float32)
    forces_std_t  = torch.tensor(np.array(forces_std),  device=device, dtype=torch.float32)

    # Accumulate per-step absolute errors
    # shape: (horizon, n_samples, 2)
    all_errors = [[] for _ in range(horizon)]

    # Accumulate unscaled ground-truth forces (all steps pooled) to compute
    # the reference standard deviation sigma for the L1/sigma normalisation.
    all_true = []

    with torch.no_grad():
        for s_t, a_seq, s_t1_seq, c_seq in dataloader_eval:
            # s_t   : (B, lookback, n_sensors)
            # a_seq : (B, lookforward, 1)   – lookforward >= horizon required
            # c_seq : (B, lookforward, 2)   – ground-truth forces per step
            B = s_t.shape[0]

            s_t   = s_t.to(device)
            a_seq = a_seq.to(device)    # (B, lookforward, 1)
            c_seq = c_seq.to(device)    # (B, lookforward, 2)

            # Encode initial latent state
            z = encoder(s_t)            # (B, latent_dim)

            for h in range(horizon):
                # One dynamics step with the ground-truth action at step h
                # a_seq may have fewer steps than horizon if lookforward < horizon
                if h < a_seq.shape[1]:
                    u_h = a_seq[:, h, :]        # (B, 1)
                else:
                    u_h = a_seq[:, -1, :]       # repeat last action

                z = dynamics(z, u_h)            # (B, latent_dim)

                # Decode forces
                c_pred_scaled = force_dec(z)    # (B, 2)
                c_pred = c_pred_scaled * forces_std_t + forces_mean_t

                # Ground-truth forces at step h
                if h < c_seq.shape[1]:
                    c_true_scaled = c_seq[:, h, :]
                else:
                    c_true_scaled = c_seq[:, -1, :]
                c_true = c_true_scaled * forces_std_t + forces_mean_t

                # Absolute error per sample
                err = torch.abs(c_pred - c_true).cpu().numpy()  # (B, 2)
                all_errors[h].append(err)

                # Store unscaled ground truth (pooled over all steps) for sigma
                all_true.append(c_true.cpu().numpy())            # (B, 2)

    # Mean MAE over all samples for each horizon step
    mae_cd = np.array([np.concatenate(all_errors[h], axis=0)[:, 0].mean()
                       for h in range(horizon)])
    mae_cl = np.array([np.concatenate(all_errors[h], axis=0)[:, 1].mean()
                       for h in range(horizon)])

    # Reference standard deviation over the whole eval set (all steps pooled),
    # in unscaled (physical) units, matching the L1/sigma metric in sec:modelPerf.
    true_all = np.concatenate(all_true, axis=0)   # (n_samples * horizon, 2)
    sigma_cd = true_all[:, 0].std()
    sigma_cl = true_all[:, 1].std()

    nmae_cd = mae_cd / sigma_cd
    nmae_cl = mae_cl / sigma_cl

    return mae_cd, mae_cl, nmae_cd, nmae_cl


# ---------------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------------

def build_tables(results, present, horizon, h25):
    """
    Build self-contained, human-readable tables of the open-loop rollout error.

    results : dict label -> (mae_cd, mae_cl, nmae_cd, nmae_cl), each array (horizon,)
    present : list of labels in display order, e.g. ['$k=3$', '$k=5$', ...]
    horizon : number of rollout steps
    h25     : the highlighted horizon step (=25) used for the manuscript numbers

    Returns a single string (also written to disk by the caller).
    """
    # Strip LaTeX math markers for plain-text headers: '$k=3$' -> 'k=3'
    def clean(lbl):
        return lbl.replace('$', '').replace('\\', '')

    names = [clean(l) for l in present]
    lines = []

    def rule(char='=', width=78):
        lines.append(char * width)

    # --- Table 1: full per-horizon error, one block per model ---------------
    rule()
    lines.append("TABLE 1 -- Open-loop rollout error per horizon step")
    lines.append("L1      : mean absolute error in physical (unscaled) units")
    lines.append("L1/sigma: L1 normalised by the std of the reference signal "
                 "on the eval set")
    lines.append("Eval set: chirp evaluation dataset")
    rule()
    for lbl in present:
        mae_cd, mae_cl, nmae_cd, nmae_cl = results[lbl]
        lines.append("")
        lines.append(f"Model {clean(lbl)}")
        header = (f"{'step h':>7} | {'Cd L1':>10} {'Cd L1/sig':>11} | "
                  f"{'Cl L1':>10} {'Cl L1/sig':>11}")
        lines.append(header)
        lines.append("-" * len(header))
        for h in range(horizon):
            star = " *" if (h + 1) == h25 else "  "
            lines.append(f"{h + 1:>7} | {mae_cd[h]:>10.5f} {nmae_cd[h]:>11.4f} | "
                         f"{mae_cl[h]:>10.5f} {nmae_cl[h]:>11.4f}{star}")
        lines.append(f"(* row marks the reported horizon step h={h25})")
    lines.append("")

    # --- Table 2: comparison across models at the reported step h=25 --------
    i25 = h25 - 1
    rule()
    lines.append(f"TABLE 2 -- Error across training horizons at step h={h25}")
    rule()
    header = (f"{'model':>8} | {'Cd L1':>10} {'Cd L1/sigma':>13} | "
              f"{'Cl L1':>10} {'Cl L1/sigma':>13}")
    lines.append(header)
    lines.append("-" * len(header))
    for lbl in present:
        mae_cd, mae_cl, nmae_cd, nmae_cl = results[lbl]
        lines.append(f"{clean(lbl):>8} | {mae_cd[i25]:>10.5f} {nmae_cd[i25]:>13.4f} | "
                     f"{mae_cl[i25]:>10.5f} {nmae_cl[i25]:>13.4f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    steps = np.arange(1, HORIZON + 1)

    results = {}
    H25 = 25  # horizon step for the reported numbers
    for label, model_id in MODEL_IDS.items():
        print(f"\nEvaluating {label}  (ID: {model_id}) ...")
        encoder, dynamics, force_dec, args = load_model(
            model_id, CHECKPOINTS_DIR, device)
        mae_cd, mae_cl, nmae_cd, nmae_cl = compute_horizon_error(
            encoder, dynamics, force_dec, args,
            EVAL_DATASET, HORIZON, device)
        results[label] = (mae_cd, mae_cl, nmae_cd, nmae_cl)

    if not results:
        print("No results to plot. Set the model IDs first.")
        return

    # --- Plot ---
    fig, (ax_cd, ax_cl) = plt.subplots(2, 1, figsize=(8, 4), sharex=True,
                                        constrained_layout=True)

    for (label, (mae_cd, mae_cl, nmae_cd, nmae_cl)), color in zip(results.items(), COLORS):
        ax_cd.plot(steps, mae_cd, color=color, lw=1.2, label=label)
        ax_cl.plot(steps, mae_cl, color=color, lw=1.2, label=label)

    ax_cd.set_ylabel(r'$L_1$ error ($C_d$)')
    ax_cl.set_ylabel(r'$L_1$ error ($C_l$)')
    ax_cl.set_xlabel(r'Prediction horizon step')
    ax_cl.set_xticks(np.arange(1, HORIZON + 1, 2))

    for ax in (ax_cd, ax_cl):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.set_xlim(1, HORIZON)
        ax.set_ylim(bottom=0)

    ax_cd.legend(loc='upper left', frameon=False, fontsize=8)

    # --- Numerical tables (printed to stdout and saved to a text file) ---
    label_order = [r'$k=3$', r'$k=5$', r'$k=10$', r'$k=25$']
    present = [lbl for lbl in label_order if lbl in results]

    table_text = build_tables(results, present, HORIZON, H25)
    print(table_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_DIR / (OUTPUT_NAME + '_tables.txt')
    with open(table_path, 'w') as f:
        f.write(table_text)
    print(f"Saved tables: {table_path}")

    for ext in ('.png', '.pdf'):
        fig.savefig((OUTPUT_DIR / OUTPUT_NAME).with_suffix(ext),
                    bbox_inches='tight')
        print(f"Saved: {(OUTPUT_DIR / OUTPUT_NAME).with_suffix(ext)}")
    plt.close(fig)


if __name__ == '__main__':
    main()