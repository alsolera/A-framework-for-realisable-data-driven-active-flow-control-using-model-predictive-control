# evaluate_ood_chirp.py
#
# Out-of-distribution evaluation on an extended-frequency chirp dataset
# (f = 0.01 to 0.80), which exceeds the training band (approx. 0.050.45).
#
# Evaluates TWO encoders on the same OOD data:
#   1. Full encoder  (90 sensors)  original checkpoint
#   2. Slim encoder  ( 4 sensors)  knowledge-distilled checkpoint
# Both share the same frozen ForceDecoder.
#
# Outputs are saved to:
#   04_Results/<case>/OOD_chirp/<BASE_MODEL_ID>/
#
# Usage:
#   python evaluate_ood_chirp.py

import json
import numpy as np
import torch
import h5py
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from libs.models import TemporalEncoder, ForceDecoder
from libs.data import get_prepared_data, loadData
from libs.test_encoder_predictor import find_model_paths, load_checkpoint
from parameters import Args as OriginalArgs

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ---------------------------------------------------------------------------
# Configuration   edit these paths/IDs as needed
# ---------------------------------------------------------------------------
BASE_MODEL_ID   = "20250906_19_38_58"
CHECKPOINTS_DIR = "03_Checkpoints"
RESULTS_DIR     = "04_Results"

# OOD dataset: extended chirp  f = 0.01 � 0.80
OOD_DATASET     = "01_Data/jet_2Dtruck_20260329_1205_oodChirpU15_f001to08_9999_no_fields.hdf5"
OOD_F_MIN       = 0.01
OOD_F_MAX       = 0.80

NUM_SENSORS_SLIM = 4     # slim encoder sensor count
CUDA             = True
BATCH_SIZE       = 512
# ---------------------------------------------------------------------------


# ============================================================
#  Data helpers
# ============================================================

def make_ood_loader(original_args, ood_path: str, device):
    """Build a dataloader for the OOD file using training scaling/config."""

    class OodArgs:
        datafile              = [ood_path]
        n_test                = 0           # use ALL data
        lookback              = original_args.lookback
        recursive_train_steps = 1           # lookforward=1 for step-by-step eval
        batch_size            = BATCH_SIZE
        augment_with_symmetry = False
        DATA_TO_GPU           = False
        sel_coefs             = ['Cd', 'Cl']

    dataloader, _ = get_prepared_data(
        OodArgs, device=torch.device('cpu'),
        shuffle_train=False, shuffle_test=False,
    )
    return dataloader, OodArgs


def load_raw_forces_ood(ood_path: str):
    """Load unscaled force coefficients directly from the OOD HDF5 file."""

    class _A:
        datafile              = [ood_path]
        augment_with_symmetry = False

    (_, forces_unscaled, *_) = loadData(_A, sel_coefs=['Cd', 'Cl'])
    return forces_unscaled          # shape (T, 2)


# ============================================================
#  Evaluation
# ============================================================

def run_evaluation(encoder, force_decoder,
                   dataloader, raw_forces_gt,
                   forces_mean, forces_std,
                   device, selected_indices=None):
    """
    Forward pass through encoder � force_decoder.

    If selected_indices is not None, only those sensor columns are fed to the
    encoder (slim mode).  Otherwise all sensors are used (full mode).

    Returns y_pred (N,2) and y_true (N,2) in physical (unscaled) units,
    plus a dict of scalar metrics.
    """
    encoder.eval()
    force_decoder.eval()

    forces_mean_np = np.array(forces_mean)
    forces_std_np  = np.array(forces_std)

    y_pred_scaled_list = []
    temp_loader = DataLoader(dataloader.dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for (s_t_full, _, _, _) in temp_loader:
            if selected_indices is not None:
                s_t = s_t_full[:, :, selected_indices].to(device)
            else:
                s_t = s_t_full.to(device)
            z_t   = encoder(s_t)
            preds = force_decoder(z_t)
            y_pred_scaled_list.append(preds.cpu())

    y_pred_scaled = torch.cat(y_pred_scaled_list).numpy()
    y_pred = y_pred_scaled * forces_std_np + forces_mean_np

    # Align ground truth: sample j in the dataloader corresponds to
    # sensor window ending at time  j + lookback - 1.
    lookback = dataloader.dataset.tensors[0].shape[1]
    offset   = lookback - 1
    n        = len(y_pred)
    y_true   = raw_forces_gt[offset: offset + n]

    if len(y_true) != n:
        raise ValueError(
            f"Alignment error: {n} predictions vs {len(y_true)} ground-truth rows."
        )

    mae      = np.mean(np.abs(y_pred - y_true), axis=0)          # (2,)
    r2       = r2_score(y_true, y_pred, multioutput='raw_values') # (2,)
    sig      = np.std(y_true, axis=0)
    norm_mae = mae / sig

    metrics = {
        'mae_cd':      float(mae[0]),
        'mae_cl':      float(mae[1]),
        'norm_mae_cd': float(norm_mae[0]),
        'norm_mae_cl': float(norm_mae[1]),
        'r2_cd':       float(r2[0]),
        'r2_cl':       float(r2[1]),
    }
    return y_pred, y_true, metrics


# ============================================================
#  Plotting helpers
# ============================================================

def _freq_axis(n, f_min=OOD_F_MIN, f_max=OOD_F_MAX):
    t = np.arange(n)
    return f_min + (f_max - f_min) * (t / max(t[-1], 1))


def plot_timeseries(y_true, y_pred_full, y_pred_slim, output_path: Path):
    """
    Four-panel figure: Cd and Cl, each showing true + full + slim predictions
    vs chirp frequency.
    """
    freqs = _freq_axis(len(y_true))
    lw    = 0.6

    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    # Training band shading
    for ax in axs:
        ax.axvspan(OOD_F_MIN, 0.05, alpha=0.08, color='tab:red',  label='_nolegend_')
        ax.axvspan(0.45,      OOD_F_MAX, alpha=0.08, color='tab:red',  label='_nolegend_')
        ax.axvspan(0.05,      0.45, alpha=0.08, color='tab:green', label='_nolegend_')

    axs[0].plot(freqs, y_true[:, 0],      color='k',          lw=lw,   label='Reference')
    axs[0].plot(freqs, y_pred_full[:, 0], color='tab:blue',   lw=lw,   linestyle='--',
                label='Full encoder (90 sensors)')
    axs[0].plot(freqs, y_pred_slim[:, 0], color='tab:orange', lw=lw,   linestyle=':',
                label='Slim encoder (4 sensors)')
    axs[0].set_ylabel('$C_d$')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    axs[1].plot(freqs, y_true[:, 1],      color='k',          lw=lw)
    axs[1].plot(freqs, y_pred_full[:, 1], color='tab:blue',   lw=lw,   linestyle='--')
    axs[1].plot(freqs, y_pred_slim[:, 1], color='tab:orange', lw=lw,   linestyle=':')
    axs[1].set_ylabel('$C_l$')
    axs[1].set_xlabel('$f_{\\mathrm{chirp}}$')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # Single shared legend (top axes)
    handles, labels = axs[0].get_legend_handles_labels()
    # Add shading proxies
    from matplotlib.patches import Patch
    handles += [
        Patch(facecolor='tab:green', alpha=0.2, label='Training band'),
        Patch(facecolor='tab:red',   alpha=0.2, label='OOD region'),
    ]
    labels += ['Training band', 'OOD region']
    axs[0].legend(handles, labels, loc='upper right', ncol=2, frameon=False,
                  fontsize='small')

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        plt.savefig(output_path.with_suffix(ext), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved time series comparison plot � {output_path.with_suffix('.png')}")


def plot_windowed_mae(y_true, y_pred_full, y_pred_slim, output_path: Path,
                     window: int = 400):
    """
    Two-panel figure showing the windowed L1/sigma error as a function of
    chirp frequency for both encoders.  Each panel is normalised by the
    global std of the respective reference signal (consistent with the
    sensor-subset evaluation figures).

    Window of ~200 steps H 8 shedding cycles (nominal period ~25 steps).
    'valid' convolution avoids edge artefacts; frequency axis trimmed to match.
    """
    from matplotlib.patches import Patch

    freqs  = _freq_axis(len(y_true))
    kernel = np.ones(window) / window

    # Global std of the reference (same normalisation as elsewhere in the paper)
    sigma = np.std(y_true, axis=0)          # (2,)  [sigma_cd, sigma_cl]

    # Normalised absolute error per sample
    err_full = np.abs(y_pred_full - y_true) / sigma   # (N, 2)
    err_slim = np.abs(y_pred_slim - y_true) / sigma   # (N, 2)

    # Windowed mean via convolution ('valid' � length N - window + 1)
    w_full_cd = np.convolve(err_full[:, 0], kernel, mode='valid')
    w_full_cl = np.convolve(err_full[:, 1], kernel, mode='valid')
    w_slim_cd = np.convolve(err_slim[:, 0], kernel, mode='valid')
    w_slim_cl = np.convolve(err_slim[:, 1], kernel, mode='valid')

    # Trim frequency axis to 'valid' output (centre-aligned)
    half      = window // 2
    freqs_win = freqs[half: half + len(w_full_cd)]

    lw = 1.2

    fig, axs = plt.subplots(2, 1, figsize=(8, 3.2), sharex=True)

    for ax in axs:
        ax.axvspan(OOD_F_MIN, 0.05,      alpha=0.08, color='tab:red')
        ax.axvspan(0.45,      OOD_F_MAX, alpha=0.08, color='tab:red')
        ax.axvspan(0.05,      0.45,      alpha=0.08, color='tab:green')
        ax.set_xlim(0, OOD_F_MAX)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # --- Cd ---
    axs[0].plot(freqs_win, w_full_cd, color='tab:blue',   lw=lw, linestyle='-',
                label='Full encoder (90 sensors)')
    axs[0].plot(freqs_win, w_slim_cd, color='tab:orange', lw=lw, linestyle='--',
                label='Slim encoder (4 sensors)')
    axs[0].set_ylabel(r'$L_1/\sigma\ (C_d)$')

    # --- Cl ---
    axs[1].plot(freqs_win, w_full_cl, color='tab:blue',   lw=lw, linestyle='-')
    axs[1].plot(freqs_win, w_slim_cl, color='tab:orange', lw=lw, linestyle='--')
    axs[1].set_ylabel(r'$L_1/\sigma\ (C_l)$')
    axs[1].set_xlabel(r'$f_{\mathrm{chirp}}$')

    # Legend above the plot region, centred, with shading proxies
    handles, labels = axs[0].get_legend_handles_labels()
    handles += [
        Patch(facecolor='tab:green', alpha=0.2, label='Training band'),
        Patch(facecolor='tab:red',   alpha=0.2, label='OOD region'),
    ]
    labels += ['Training band', 'OOD region']
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False, fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ('.png', '.pdf'):
        plt.savefig(output_path.with_suffix(ext), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved windowed L1/sigma plot � {output_path.with_suffix('.png')}")


def plot_scatter(y_true, y_pred, encoder_label: str, output_path: Path):
    """Scatter plot (true vs predicted) coloured by chirp frequency."""
    freqs = _freq_axis(len(y_true))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    for ax, col_idx, xlabel, ylabel in [
        (ax1, 0, 'Reference $C_d$',  f'Predicted $\\hat{{C}}_d$'),
        (ax2, 1, 'Reference $C_l$',  f'Predicted $\\hat{{C}}_l$'),
    ]:
        sc = ax.scatter(y_true[:, col_idx], y_pred[:, col_idx],
                        c=freqs, cmap='viridis', s=1, alpha=0.6, rasterized=True)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.7)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('$f_{\\mathrm{chirp}}$')
    cbar.outline.set_visible(False)

    fig.suptitle(encoder_label, y=1.01, fontsize='medium')

    for ext in ('.png', '.pdf'):
        plt.savefig(output_path.with_suffix(ext), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"Saved scatter plot � {output_path.with_suffix('.png')}")


def plot_metrics_bar(metrics_full, metrics_slim, output_path: Path):
    """Side-by-side bar chart comparing norm_MAE and R2 for both encoders."""
    labels    = ['Full (90)', 'Slim (4)']
    nmae_cd   = [metrics_full['norm_mae_cd'], metrics_slim['norm_mae_cd']]
    nmae_cl   = [metrics_full['norm_mae_cl'], metrics_slim['norm_mae_cl']]
    r2_cd     = [metrics_full['r2_cd'],       metrics_slim['r2_cd']]
    r2_cl     = [metrics_full['r2_cl'],       metrics_slim['r2_cl']]

    x   = np.arange(len(labels))
    w   = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    ax1.bar(x - w/2, nmae_cd, w, label='$C_d$', color='tab:blue',   alpha=0.8)
    ax1.bar(x + w/2, nmae_cl, w, label='$C_l$', color='tab:orange', alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel(r'Relative error ($L_1/\sigma$)')
    ax1.set_title('Normalised MAE')
    ax1.legend(frameon=False)
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, size='large')

    ax2.bar(x - w/2, r2_cd, w, label='$C_d$', color='tab:blue',   alpha=0.8)
    ax2.bar(x + w/2, r2_cl, w, label='$C_l$', color='tab:orange', alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel('$R^2$')
    ax2.set_title('$R^2$ score')
    ax2.legend(frameon=False)
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, size='large')

    plt.tight_layout()
    for ext in ('.png', '.pdf'):
        plt.savefig(output_path.with_suffix(ext), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved metrics bar chart � {output_path.with_suffix('.png')}")


# ============================================================
#  Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Locate model artefacts and load training config / scaling
    # ------------------------------------------------------------------
    base_ckp_dir, base_results_dir = find_model_paths(BASE_MODEL_ID, CHECKPOINTS_DIR)
    if base_ckp_dir is None:
        raise RuntimeError("Could not locate model paths. Check BASE_MODEL_ID / CHECKPOINTS_DIR.")

    latent_file = next(
        f for f in sorted(base_results_dir.glob(f"{BASE_MODEL_ID}*latent_space.hdf5"))
        if '_eval' not in f.name
    )
    with h5py.File(latent_file, 'r') as fh:
        original_args = OriginalArgs(**json.loads(fh.attrs['args']))
        forces_mean   = fh['forces_mean'][:]
        forces_std    = fh['forces_std'][:]
    print(f"Loaded training config from: {latent_file}")

    # ------------------------------------------------------------------
    # 2. Resolve OOD dataset path and build dataloader
    # ------------------------------------------------------------------
    project_root = Path(CHECKPOINTS_DIR).parent
    ood_path     = str(project_root / OOD_DATASET)
    if not Path(ood_path).exists():
        raise FileNotFoundError(f"OOD dataset not found at: {ood_path}")
    print(f"OOD dataset: {ood_path}")

    dataloader_ood, _ = make_ood_loader(original_args, ood_path, device)
    print(f"OOD dataloader: {len(dataloader_ood.dataset)} samples "
          f"(lookback={original_args.lookback})")

    raw_forces_ood = load_raw_forces_ood(ood_path)
    print(f"OOD raw forces shape: {raw_forces_ood.shape}")

    # ------------------------------------------------------------------
    # 3. Output directory
    # ------------------------------------------------------------------
    output_dir = Path(RESULTS_DIR) / original_args.case / "OOD_chirp" / BASE_MODEL_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # ------------------------------------------------------------------
    # 4. Load ForceDecoder (shared by both encoders)
    # ------------------------------------------------------------------
    force_decoder = ForceDecoder(
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.forces_hidden_dim,
        arch=original_args.force_decoder_arch,
    ).to(device)
    force_ckpt = base_ckp_dir / f"{original_args.modelname}_force.pth.tar"
    load_checkpoint(force_decoder, str(force_ckpt), device)
    force_decoder.eval()
    print(f"Loaded ForceDecoder from: {force_ckpt}")

    # ------------------------------------------------------------------
    # 5. Full encoder (90 sensors)
    # ------------------------------------------------------------------
    n_sensors_full = dataloader_ood.dataset.tensors[0].shape[-1]  # infer from data
    full_encoder = TemporalEncoder(
        input_dim=n_sensors_full,
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim,
        num_layers=original_args.n_layers,
    ).to(device)
    full_ckpt = base_ckp_dir / f"{original_args.modelname}_encoder.pth.tar"
    load_checkpoint(full_encoder, str(full_ckpt), device)
    full_encoder.eval()
    print(f"Loaded full encoder ({n_sensors_full} sensors) from: {full_ckpt}")

    y_pred_full, y_true, metrics_full = run_evaluation(
        full_encoder, force_decoder,
        dataloader_ood, raw_forces_ood,
        forces_mean, forces_std, device,
        selected_indices=None,
    )
    print("\n[Full encoder] OOD metrics:")
    for k, v in metrics_full.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 6. Slim encoder (4 sensors)
    # ------------------------------------------------------------------
    # Recover the same SHAP-ranked sensor indices used during training
    base_artifact_dir = (
        Path(RESULTS_DIR) / original_args.case /
        "sensor_selection" / BASE_MODEL_ID
    )
    shap_path = base_artifact_dir / "mean_abs_shap_values.txt"
    if not shap_path.exists():
        raise FileNotFoundError(
            f"SHAP values not found at {shap_path}.\n"
            "Run rank_sensors_shap_and_train_slim.py first."
        )
    mean_abs_shap   = np.loadtxt(shap_path)
    sensor_ranking  = np.argsort(mean_abs_shap)[::-1]
    selected_indices = np.sort(sensor_ranking[:NUM_SENSORS_SLIM])
    print(f"\nSlim encoder selected sensor indices: {selected_indices}")

    slim_encoder = TemporalEncoder(
        input_dim=NUM_SENSORS_SLIM,
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim,
        num_layers=original_args.n_layers,
    ).to(device)
    slim_ckpt = (
        base_artifact_dir / "subset_evaluation" /
        f"slim_encoder_{NUM_SENSORS_SLIM}sensors.pth.tar"
    )
    load_checkpoint(slim_encoder, str(slim_ckpt), device)
    slim_encoder.eval()
    print(f"Loaded slim encoder from: {slim_ckpt}")

    y_pred_slim, _, metrics_slim = run_evaluation(
        slim_encoder, force_decoder,
        dataloader_ood, raw_forces_ood,
        forces_mean, forces_std, device,
        selected_indices=selected_indices,
    )
    print("\n[Slim encoder] OOD metrics:")
    for k, v in metrics_slim.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # 7. Save metrics to JSON
    # ------------------------------------------------------------------
    results = {
        'ood_dataset':   OOD_DATASET,
        'f_min':         OOD_F_MIN,
        'f_max':         OOD_F_MAX,
        'full_encoder':  metrics_full,
        'slim_encoder':  metrics_slim,
    }
    json_path = output_dir / "ood_metrics.json"
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=4)
    print(f"\nSaved metrics to: {json_path}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    plot_timeseries(
        y_true, y_pred_full, y_pred_slim,
        output_dir / "ood_timeseries_comparison",
    )

    plot_scatter(
        y_true, y_pred_full,
        encoder_label="Full encoder (90 sensors)",
        output_path=output_dir / "ood_scatter_full",
    )
    plot_scatter(
        y_true, y_pred_slim,
        encoder_label=f"Slim encoder ({NUM_SENSORS_SLIM} sensors)",
        output_path=output_dir / "ood_scatter_slim",
    )

    plot_windowed_mae(
        y_true, y_pred_full, y_pred_slim,
        output_dir / "ood_windowed_mae",
    )

    plot_metrics_bar(
        metrics_full, metrics_slim,
        output_dir / "ood_metrics_bar",
    )

    print("\nDone. All OOD results saved to:", output_dir)


if __name__ == "__main__":
    main()