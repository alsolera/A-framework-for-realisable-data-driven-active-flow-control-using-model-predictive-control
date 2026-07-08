# evaluate_noise_robustness.py
#
# Evaluates the robustness of the 4-sensor slim encoder to additive Gaussian
# sensor noise. Noise is injected at test time only (no retraining).
#
# Noise characterisation: for each noise level alpha, the standard deviation
# of the injected noise is:
#       sigma_noise = alpha
# This gives a single, physically meaningful noise
# scale common to all sensors.
#
# Outputs (saved to the subset_evaluation directory alongside existing results):
#   - noise_robustness_results.json   -- metrics per noise level
#   - noise_robustness.png/pdf/eps    -- normalised MAE and R2 vs noise level

import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from libs.models import TemporalEncoder, ForceDecoder
from libs.data import prepare_raw_data, build_dataloader_from_scaled, loadData
from libs.test_encoder_predictor import find_model_paths, load_checkpoint
from parameters import Args as OriginalArgs
import h5py

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL_ID      = "20250906_19_38_58"
CHECKPOINTS_DIR    = "03_Checkpoints"
RESULTS_DIR        = "04_Results"
NUM_SENSORS        = 4          # slim encoder to evaluate
N_SENSORS_FULL     = 90         # total sensors in the full model

# Noise levels: sigma_noise = alpha * global_std
# where global_std = std of ALL 90 normalised sensor readings (scalar, same as
# used for data normalisation in loadData). Consistent across all three scripts.
NOISE_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

SEED = 4321
CUDA = True
# ---------------------------------------------------------------------------


def load_raw_forces(args_for_data, sel_coefs=('Cd', 'Cl')):
    """Load full unscaled force data from the evaluation dataset."""
    class _Args:
        datafile = args_for_data.datafile
        augment_with_symmetry = False
    (_, forces_unscaled, *_) = loadData(_Args, sel_coefs=list(sel_coefs))
    return forces_unscaled


def evaluate_force_prediction(
        slim_encoder: TemporalEncoder,
        force_decoder: ForceDecoder,
        dataloader,
        selected_indices: np.ndarray,
        raw_forces_gt: np.ndarray,
        forces_mean: np.ndarray,
        forces_std: np.ndarray,
        device: torch.device,
):
    """
    Clean forward pass — noise has already been baked into the dataloader.
    Returns norm_mae_cd, norm_mae_cl, r2_cd, r2_cl.
    """
    slim_encoder.eval()
    force_decoder.eval()

    y_pred_scaled_list = []
    temp_loader = DataLoader(dataloader.dataset, batch_size=512, shuffle=False)

    with torch.no_grad():
        for (s_t_full, _, _, _) in temp_loader:
            s_t_slim     = s_t_full[:, :, selected_indices].to(device)
            z_t          = slim_encoder(s_t_slim)
            preds_scaled = force_decoder(z_t)
            y_pred_scaled_list.append(preds_scaled.cpu())

    y_pred_scaled = torch.cat(y_pred_scaled_list).numpy()
    y_pred = y_pred_scaled * forces_std + forces_mean

    lookback = dataloader.dataset.tensors[0].shape[1]
    offset   = lookback - 1
    n        = len(y_pred)
    y_true   = raw_forces_gt[offset: offset + n]

    if len(y_true) != n:
        raise ValueError(f"Alignment mismatch: {n} predictions vs {len(y_true)} ground-truth rows.")

    mae_cd = float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0])))
    mae_cl = float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1])))
    r2     = r2_score(y_true, y_pred, multioutput='raw_values')

    forces_std_test = np.std(y_true, axis=0)
    norm_mae_cd = mae_cd / forces_std_test[0]
    norm_mae_cl = mae_cl / forces_std_test[1]

    return norm_mae_cd, norm_mae_cl, float(r2[0]), float(r2[1])


def _add_secondary_xaxis(ax, sensor_std_factor):
    """Add a top x-axis showing noise as % of the 4 selected sensors' mean std."""
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / sensor_std_factor)
    ax2.set_xlabel(r'Gaussian noise $\sigma$ (\% of selected sensors $\sigma$)',
                   fontsize='small')
    ax2.tick_params(axis='x', labelsize='small')
    return ax2


def plot_results(results, output_path: Path, sensor_std_factor: float):
    """Two-panel figure: normalised MAE and R2 vs noise level (alpha in %)."""
    alphas_pct  = [r['noise_alpha_pct'] for r in results]
    nmae_cd     = [r['norm_mae_cd']     for r in results]
    nmae_cl     = [r['norm_mae_cl']     for r in results]
    r2_cd       = [r['r2_cd']           for r in results]
    r2_cl       = [r['r2_cl']           for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.4))

    # --- Subplot (a): normalised MAE ---
    ax1.plot(alphas_pct, nmae_cd, 'o-', color='tab:blue',   markersize=5, label=r'$C_d$')
    ax1.plot(alphas_pct, nmae_cl, 's-', color='tab:orange', markersize=5, label=r'$C_l$')
    ax1.set_xlabel(r'Gaussian noise $\sigma$ (\% of global $\sigma$)')
    ax1.set_ylabel(r'Relative error ($L_1/\sigma$)')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.15, 1.18, '(a)', transform=ax1.transAxes, size='large')
    _add_secondary_xaxis(ax1, sensor_std_factor)

    # --- Subplot (b): R2 ---
    ax2.plot(alphas_pct, r2_cd, 'o-', color='tab:blue',   markersize=5, label=r'$C_d$')
    ax2.plot(alphas_pct, r2_cl, 's-', color='tab:orange', markersize=5, label=r'$C_l$')
    ax2.set_xlabel(r'Gaussian noise $\sigma$ (\% of global $\sigma$)')
    ax2.set_ylabel(r'$R^2$')
    if ax2.get_ylim()[0] < 0:
        ax2.set_ylim(bottom=0.0)
    ax2.set_ylim(top=0.95)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.15, 1.18, '(b)', transform=ax2.transAxes, size='large')
    _add_secondary_xaxis(ax2, sensor_std_factor)

    # Shared legend below the top axes
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.eps'), bbox_inches='tight')
    print(f"Saved noise robustness plot to {output_path}")
    plt.close(fig)


def main():
    # --- Setup ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng    = np.random.default_rng(SEED)   # passed to build_dataloader_from_scaled for noise
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    print(f"Using device: {device}")

    # --- Locate model artefacts ---
    base_ckp_dir, base_results_dir = find_model_paths(BASE_MODEL_ID, CHECKPOINTS_DIR)
    if base_ckp_dir is None:
        raise RuntimeError("Could not find model paths. Check BASE_MODEL_ID and CHECKPOINTS_DIR.")

    # Load original args from the latent space file
    latent_file = next(
        f for f in sorted(base_results_dir.glob(f"{BASE_MODEL_ID}*latent_space.hdf5"))
        if '_eval' not in f.name
    )
    with h5py.File(latent_file, 'r') as f:
        original_args = OriginalArgs(**json.loads(f.attrs['args']))
        forces_mean = f['forces_mean'][:]
        forces_std  = f['forces_std'][:]
    print(f"Loaded model config from {latent_file}")

    # Output directory: same as subset_evaluation for consistency
    base_artifact_dir = (
        Path(RESULTS_DIR) / original_args.case /
        "sensor_selection" / BASE_MODEL_ID
    )
    output_dir = base_artifact_dir / "subset_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load SHAP ranking to recover the same 4 sensor indices ---
    shap_path = base_artifact_dir / "mean_abs_shap_values.txt"
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP values not found at {shap_path}. Run rank_sensors_shap_and_train_slim.py first.")
    mean_abs_shap = np.loadtxt(shap_path)
    sensor_ranking = np.argsort(mean_abs_shap)[::-1]
    selected_indices = np.sort(sensor_ranking[:NUM_SENSORS])
    print(f"Selected sensor indices (top {NUM_SENSORS} by SHAP): {selected_indices}")

    # --- Load slim encoder and force decoder ---
    slim_ckpt_path = output_dir / f"slim_encoder_{NUM_SENSORS}sensors.pth.tar"
    if not slim_ckpt_path.exists():
        raise FileNotFoundError(
            f"Slim encoder checkpoint not found at {slim_ckpt_path}.\n"
            f"Run evaluate_sensor_subsets.py first to train the slim encoder."
        )

    slim_encoder = TemporalEncoder(
        input_dim=NUM_SENSORS,
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim,
        num_layers=original_args.n_layers
    ).to(device)
    load_checkpoint(slim_encoder, str(slim_ckpt_path), device)
    slim_encoder.eval()
    print(f"Loaded slim encoder from {slim_ckpt_path}")

    force_decoder = ForceDecoder(
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.forces_hidden_dim,
        arch=original_args.force_decoder_arch
    ).to(device)
    force_ckpt_path = base_ckp_dir / f"{original_args.modelname}_force.pth.tar"
    load_checkpoint(force_decoder, str(force_ckpt_path), device)
    force_decoder.eval()
    print(f"Loaded force decoder from {force_ckpt_path}")

    # --- Load and scale eval data ONCE ---
    loader_args_eval = deepcopy(original_args)
    project_root = Path(CHECKPOINTS_DIR).parent
    eval_paths_raw = original_args.eval_dataset
    if isinstance(eval_paths_raw, str):
        eval_paths_raw = [eval_paths_raw]
    loader_args_eval.datafile           = [str(project_root / p) for p in eval_paths_raw]
    loader_args_eval.batch_size         = 512
    loader_args_eval.n_test             = 0
    loader_args_eval.augment_with_symmetry = False

    p_scaled_eval, control_scaled_eval, forces_scaled_eval, file_end_indices_eval = (
        prepare_raw_data(loader_args_eval))
    raw_forces_gt = load_raw_forces(loader_args_eval)
    print(f"Loaded evaluation dataset: {p_scaled_eval.shape[0]} timesteps, "
          f"forces shape: {raw_forces_gt.shape}")

    # Mean std of the 4 selected sensors in normalised space.
    # Since global p_std > 1 for these sensors, their normalised std > 1.
    # The secondary x-axis shows noise as % of selected-sensor std = primary / sensor_std_factor.
    sensor_std_factor = float(np.std(p_scaled_eval[:, selected_indices], axis=0).mean())
    print(f"Mean std of 4 selected sensors (normalised): {sensor_std_factor:.4f}")

    # --- Main loop: build a fresh noisy dataloader per alpha level ---
    results = []
    for alpha in NOISE_ALPHAS:
        noise_sigma = alpha      # data is z-scored; alpha is directly noise/signal ratio
        print(f"\nNoise alpha={alpha*100:.3f}%  =>  sigma_noise={noise_sigma:.6f}")

        dl_noisy, _ = build_dataloader_from_scaled(
            p_scaled_eval, control_scaled_eval, forces_scaled_eval,
            file_end_indices_eval, loader_args_eval, device,
            noise_sigma=noise_sigma, rng=rng, shuffle_train=False)

        norm_mae_cd, norm_mae_cl, r2_cd, r2_cl = evaluate_force_prediction(
            slim_encoder, force_decoder, dl_noisy,
            selected_indices, raw_forces_gt,
            forces_mean, forces_std, device
        )
        print(f"  norm_MAE: Cd={norm_mae_cd:.4f}, Cl={norm_mae_cl:.4f} | "
              f"R2: Cd={r2_cd:.4f}, Cl={r2_cl:.4f}")

        results.append({
            'noise_alpha':     alpha,
            'noise_alpha_pct': alpha * 100,
            'noise_sigma':     noise_sigma,
            'norm_mae_cd':     norm_mae_cd,
            'norm_mae_cl':     norm_mae_cl,
            'r2_cd':           r2_cd,
            'r2_cl':           r2_cl,
        })

    # --- Save results ---
    results_file = output_dir / "noise_robustness_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {results_file}")

    # --- Plot ---
    plot_path = output_dir / "noise_robustness.png"
    plot_results(results, plot_path, sensor_std_factor)

    print("\nDone.")


if __name__ == "__main__":
    main()