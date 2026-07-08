# evaluate_noise_robustness_augmented.py
#
# For each training noise level alpha_train (excluding 0, which reuses the
# existing clean checkpoint), trains a 4-sensor slim encoder from scratch with
# additive Gaussian noise injected on the sensor inputs during training.
# Each noise-trained model is then evaluated at its own alpha_train level.
#
# The final combined plot overlays:
#   - Clean model degradation curve  (loaded from noise_robustness_results.json,
#     produced by evaluate_noise_robustness.py run that first)
#   - Noise-augmented model curve    (one point per alpha_train, evaluated at
#     the matching alpha level)
#
# Outputs (saved to subset_evaluation/noise_augmented/):
#   - slim_encoder_4sensors_noise{alpha_pct:.0f}pct.pth.tar  -- one ckpt per alpha
#   - noise_augmented_results.json                            -- metrics per alpha
#   - noise_robustness_combined.png/pdf/eps                   -- combined comparison plot

import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

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
# Configuration  keep NOISE_ALPHAS identical to evaluate_noise_robustness.py
# ---------------------------------------------------------------------------
BASE_MODEL_ID   = "20250906_19_38_58"
CHECKPOINTS_DIR = "03_Checkpoints"
RESULTS_DIR     = "04_Results"
NUM_SENSORS     = 4

# Must match evaluate_noise_robustness.py exactly so axes align in the plot
NOISE_ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

# Slim encoder training hyperparameters (mirror evaluate_sensor_subsets.py)
SLIM_EPOCHS    = 100
SLIM_LR        = 1e-3
SLIM_BATCH     = 256

SEED = 4321
CUDA = True
# ---------------------------------------------------------------------------


def load_raw_forces(args_for_data, sel_coefs=('Cd', 'Cl')):
    class _Args:
        datafile = args_for_data.datafile
        augment_with_symmetry = False
    (_, forces_unscaled, *_) = loadData(_Args, sel_coefs=list(sel_coefs))
    return forces_unscaled


def train_slim_encoder_with_noise(
        slim_encoder: TemporalEncoder,
        encoder_orig: TemporalEncoder,
        p_scaled_train: np.ndarray,
        control_scaled_train: np.ndarray,
        forces_scaled_train: np.ndarray,
        file_end_indices_train: list,
        loader_args_train,
        selected_indices: np.ndarray,
        noise_sigma: float,
        device: torch.device,
        checkpoint_path: Path,
        rng: np.random.Generator,
):
    """
    Train a slim encoder to mimic the full encoder's latent space.
    Each epoch, fresh noise is added to the full sensor time series BEFORE
    sequence creation  so all sequences sharing timestep t see the same
    noise realisation for t.
    Validation uses clean inputs for fair early stopping.
    """
    optimizer     = optim.Adam(slim_encoder.parameters(), lr=SLIM_LR)
    best_val_loss = float('inf')

    # Build a clean validation loader once
    _, dl_val = build_dataloader_from_scaled(
        p_scaled_train, control_scaled_train, forces_scaled_train,
        file_end_indices_train, loader_args_train, device,
        noise_sigma=0.0, shuffle_train=False)

    pbar = tqdm(range(1, SLIM_EPOCHS + 1),
                desc=f"Training noise-augmented slim encoder (sigma={noise_sigma:.5f})",
                leave=False)

    for epoch in pbar:
        # Fresh noisy dataloader each epoch
        dl_train_noisy, _ = build_dataloader_from_scaled(
            p_scaled_train, control_scaled_train, forces_scaled_train,
            file_end_indices_train, loader_args_train, device,
            noise_sigma=noise_sigma, rng=rng, shuffle_train=True)

        slim_encoder.train()
        for s_t_full, _, _, _ in dl_train_noisy:
            s_t_full = s_t_full.to(device)
            s_t_slim = s_t_full[:, :, selected_indices]
            optimizer.zero_grad()
            with torch.no_grad():
                z_target = encoder_orig(s_t_full)
            loss = F.mse_loss(slim_encoder(s_t_slim), z_target)
            loss.backward()
            optimizer.step()

        # Clean validation
        slim_encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s_t_full_v, _, _, _ in dl_val:
                s_t_slim_v = s_t_full_v[:, :, selected_indices].to(device)
                z_tgt_v    = encoder_orig(s_t_full_v.to(device))
                val_loss  += F.mse_loss(slim_encoder(s_t_slim_v), z_tgt_v).item()

        avg_val = val_loss / len(dl_val)
        pbar.set_postfix({'best_val': f'{best_val_loss:.6f}'})
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'state_dict': slim_encoder.state_dict()}, checkpoint_path)

    print(f"  Training done. Best val loss: {best_val_loss:.6f} -> {checkpoint_path}")


def evaluate_force_prediction(
        slim_encoder: TemporalEncoder,
        force_decoder: ForceDecoder,
        dataloader: DataLoader,
        selected_indices: np.ndarray,
        raw_forces_gt: np.ndarray,
        forces_mean: np.ndarray,
        forces_std: np.ndarray,
        device: torch.device,
):
    """Clean forward pass  noise already baked into the dataloader."""
    slim_encoder.eval()
    force_decoder.eval()
    preds_list = []
    with torch.no_grad():
        for (s_t_full, _, _, _) in DataLoader(dataloader.dataset, batch_size=512, shuffle=False):
            s_t = s_t_full[:, :, selected_indices].to(device)
            preds_list.append(force_decoder(slim_encoder(s_t)).cpu())

    y_pred = torch.cat(preds_list).numpy() * forces_std + forces_mean
    offset = dataloader.dataset.tensors[0].shape[1] - 1
    y_true = raw_forces_gt[offset: offset + len(y_pred)]
    if len(y_true) != len(y_pred):
        raise ValueError(f"Alignment mismatch: {len(y_pred)} preds vs {len(y_true)} GT rows.")

    std_test    = np.std(y_true, axis=0)
    mae_cd      = float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0])))
    mae_cl      = float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1])))
    r2          = r2_score(y_true, y_pred, multioutput='raw_values')
    return mae_cd / std_test[0], mae_cl / std_test[1], float(r2[0]), float(r2[1])


def _add_secondary_xaxis(ax, sensor_std_factor):
    """Add a top x-axis showing noise as % of the 4 selected sensors' mean std."""
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / sensor_std_factor)
    ax2.set_xlabel(r'Gaussian noise $\sigma$ (\% of selected sensors $\sigma$)',
                   fontsize='small')
    ax2.tick_params(axis='x', labelsize='small')
    return ax2


def plot_combined(clean_results, aug_results, output_path: Path, sensor_std_factor: float):
    """
    Overlay clean-model degradation curve and noise-augmented model curve.
    Both share the same x-axis (alpha in %).
    """
    # Clean model: full sweep
    c_alpha  = [r['noise_alpha_pct'] for r in clean_results]
    c_ncd    = [r['norm_mae_cd']     for r in clean_results]
    c_ncl    = [r['norm_mae_cl']     for r in clean_results]
    c_r2cd   = [r['r2_cd']           for r in clean_results]
    c_r2cl   = [r['r2_cl']           for r in clean_results]

    # Augmented model: one point per trained alpha (evaluated at same alpha)
    a_alpha  = [r['noise_alpha_pct'] for r in aug_results]
    a_ncd    = [r['norm_mae_cd']     for r in aug_results]
    a_ncl    = [r['norm_mae_cl']     for r in aug_results]
    a_r2cd   = [r['r2_cd']           for r in aug_results]
    a_r2cl   = [r['r2_cl']           for r in aug_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.4))

    # Colours: blue=Cd, orange=Cl; solid=clean, dashed=augmented
    kw_clean = dict(markersize=5, linewidth=1.2)
    kw_aug   = dict(markersize=5, linewidth=1.2, linestyle='--')

    ax1.plot(c_alpha, c_ncd,  'o-',  color='tab:blue',   label=r'$C_d$ clean',    **kw_clean)
    ax1.plot(c_alpha, c_ncl,  's-',  color='tab:orange', label=r'$C_l$ clean',    **kw_clean)
    ax1.plot(a_alpha, a_ncd,  'o--', color='tab:blue',   label=r'$C_d$ aug.',     **kw_aug)
    ax1.plot(a_alpha, a_ncl,  's--', color='tab:orange', label=r'$C_l$ aug.',     **kw_aug)
    ax1.set_xlabel(r'Gaussian noise $\sigma$ (\% of global $\sigma$)')
    ax1.set_ylabel(r'Relative error ($L_1/\sigma$)')
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.15, 1.18, '(a)', transform=ax1.transAxes, size='large')
    _add_secondary_xaxis(ax1, sensor_std_factor)

    ax2.plot(c_alpha, c_r2cd, 'o-',  color='tab:blue',   label=r'$C_d$ clean',   **kw_clean)
    ax2.plot(c_alpha, c_r2cl, 's-',  color='tab:orange', label=r'$C_l$ clean',   **kw_clean)
    ax2.plot(a_alpha, a_r2cd, 'o--', color='tab:blue',   label=r'$C_d$ aug.',    **kw_aug)
    ax2.plot(a_alpha, a_r2cl, 's--', color='tab:orange', label=r'$C_l$ aug.',    **kw_aug)
    ax2.set_xlabel(r'Gaussian noise $\sigma$ (\% of global $\sigma$)')
    ax2.set_ylabel(r'$R^2$')
    if ax2.get_ylim()[0] < 0:
        ax2.set_ylim(bottom=0.0)
    ax2.set_ylim(top=0.95)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.15, 1.18, '(b)', transform=ax2.transAxes, size='large')
    _add_secondary_xaxis(ax2, sensor_std_factor)

    # Single shared legend at the bottom
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False, fontsize='small',
               handlelength=3.5, handletextpad=0.5, columnspacing=1.2)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.eps'), bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")
    plt.close(fig)


def main():
    # --- Setup ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA else 'cpu')
    print(f"Using device: {device}")

    # --- Locate model artefacts ---
    base_ckp_dir, base_results_dir = find_model_paths(BASE_MODEL_ID, CHECKPOINTS_DIR)
    if base_ckp_dir is None:
        raise RuntimeError("Could not find model paths. Check BASE_MODEL_ID / CHECKPOINTS_DIR.")

    latent_file = next(
        f for f in sorted(base_results_dir.glob(f"{BASE_MODEL_ID}*latent_space.hdf5"))
        if '_eval' not in f.name
    )
    with h5py.File(latent_file, 'r') as f:
        original_args = OriginalArgs(**json.loads(f.attrs['args']))
        forces_mean   = f['forces_mean'][:]
        forces_std    = f['forces_std'][:]
    print(f"Loaded model config from {latent_file}")

    # --- Output directories ---
    base_artifact_dir = (
        Path(RESULTS_DIR) / original_args.case /
        "sensor_selection" / BASE_MODEL_ID
    )
    subset_eval_dir = base_artifact_dir / "subset_evaluation"
    output_dir      = subset_eval_dir / "noise_augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load clean-model results (must exist from evaluate_noise_robustness.py) ---
    clean_results_path = subset_eval_dir / "noise_robustness_results.json"
    if not clean_results_path.exists():
        raise FileNotFoundError(
            f"Clean model results not found at {clean_results_path}.\n"
            "Run evaluate_noise_robustness.py first."
        )
    with open(clean_results_path) as f:
        clean_results = json.load(f)
    print(f"Loaded clean model results ({len(clean_results)} points).")

    # --- SHAP ranking -> selected sensor indices ---
    shap_path = base_artifact_dir / "mean_abs_shap_values.txt"
    if not shap_path.exists():
        raise FileNotFoundError(f"SHAP values not found at {shap_path}.")
    mean_abs_shap    = np.loadtxt(shap_path)
    sensor_ranking   = np.argsort(mean_abs_shap)[::-1]
    selected_indices = np.sort(sensor_ranking[:NUM_SENSORS])
    print(f"Selected sensor indices (top {NUM_SENSORS} by SHAP): {selected_indices}")

    # --- Load full encoder (used as distillation target during training) ---
    encoder_orig = TemporalEncoder(
        input_dim=mean_abs_shap.shape[0],
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim,
        num_layers=original_args.n_layers
    ).to(device)
    enc_path = base_ckp_dir / f"{original_args.modelname}_encoder.pth.tar"
    load_checkpoint(encoder_orig, str(enc_path), device)
    encoder_orig.eval()

    # --- Load force decoder ---
    force_decoder = ForceDecoder(
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.forces_hidden_dim,
        arch=original_args.force_decoder_arch
    ).to(device)
    force_path = base_ckp_dir / f"{original_args.modelname}_force.pth.tar"
    load_checkpoint(force_decoder, str(force_path), device)
    force_decoder.eval()

    # --- Load and scale training data ONCE ---
    loader_args_train              = deepcopy(original_args)
    loader_args_train.batch_size   = SLIM_BATCH
    loader_args_train.n_test       = 5000
    loader_args_train.augment_with_symmetry = False
    p_scaled_train, control_scaled_train, forces_scaled_train, file_end_indices_train = (
        prepare_raw_data(loader_args_train))

    # --- Load and scale eval data ONCE ---
    loader_args_eval               = deepcopy(original_args)
    project_root                   = Path(CHECKPOINTS_DIR).parent
    eval_paths                     = original_args.eval_dataset
    if isinstance(eval_paths, str):
        eval_paths = [eval_paths]
    loader_args_eval.datafile           = [str(project_root / p) for p in eval_paths]
    loader_args_eval.batch_size         = 512
    loader_args_eval.n_test             = 0
    loader_args_eval.augment_with_symmetry = False
    p_scaled_eval, control_scaled_eval, forces_scaled_eval, file_end_indices_eval = (
        prepare_raw_data(loader_args_eval))
    raw_forces_gt = load_raw_forces(loader_args_eval)

    sensor_std_factor = float(np.std(p_scaled_eval[:, selected_indices], axis=0).mean())
    print(f"Mean std of 4 selected sensors (normalised): {sensor_std_factor:.4f}")

    # --- Main loop: train one slim encoder per non-zero alpha, evaluate at same alpha ---
    aug_results      = []
    training_alphas  = [a for a in NOISE_ALPHAS if a > 0.0]

    for alpha in training_alphas:
        noise_sigma = alpha      # data is z-scored; alpha is directly noise/signal ratio
        alpha_pct   = alpha * 100
        print(f"\n{'='*60}")
        print(f"Alpha = {alpha_pct:.3f}%  |  sigma_noise = {noise_sigma:.6f}")
        print(f"{'='*60}")

        ckpt_path = output_dir / f"slim_encoder_{NUM_SENSORS}sensors_noise{alpha_pct:.3f}pct.pth.tar"

        slim_encoder = TemporalEncoder(
            input_dim=NUM_SENSORS,
            latent_dim=original_args.latent_dim,
            hidden_dim=original_args.enc_hidden_dim,
            num_layers=original_args.n_layers
        ).to(device)

        if ckpt_path.exists():
            print(f"  Found existing checkpoint, skipping training.")
        else:
            train_slim_encoder_with_noise(
                slim_encoder, encoder_orig,
                p_scaled_train, control_scaled_train, forces_scaled_train,
                file_end_indices_train, loader_args_train,
                selected_indices, noise_sigma,
                device, ckpt_path, rng
            )

        load_checkpoint(slim_encoder, str(ckpt_path), device)
        slim_encoder.eval()

        # Evaluate at matching noise level
        dl_noisy, _ = build_dataloader_from_scaled(
            p_scaled_eval, control_scaled_eval, forces_scaled_eval,
            file_end_indices_eval, loader_args_eval, device,
            noise_sigma=noise_sigma, rng=rng, shuffle_train=False)

        norm_mae_cd, norm_mae_cl, r2_cd, r2_cl = evaluate_force_prediction(
            slim_encoder, force_decoder, dl_noisy,
            selected_indices, raw_forces_gt,
            forces_mean, forces_std, device)
        print(f"  norm_MAE: Cd={norm_mae_cd:.4f}, Cl={norm_mae_cl:.4f} | "
              f"R2: Cd={r2_cd:.4f}, Cl={r2_cl:.4f}")

        aug_results.append({
            'noise_alpha':     alpha,
            'noise_alpha_pct': alpha_pct,
            'noise_sigma':     noise_sigma,
            'norm_mae_cd':     norm_mae_cd,
            'norm_mae_cl':     norm_mae_cl,
            'r2_cd':           r2_cd,
            'r2_cl':           r2_cl,
        })

    # --- Save augmented results ---
    aug_results_file = output_dir / "noise_augmented_results.json"
    with open(aug_results_file, 'w') as f:
        json.dump(aug_results, f, indent=4)
    print(f"\nSaved augmented results to {aug_results_file}")

    # --- Combined plot ---
    combined_plot_path = subset_eval_dir / "noise_robustness_combined.png"
    plot_combined(clean_results, aug_results, combined_plot_path, sensor_std_factor)

    print("\nDone.")


if __name__ == "__main__":
    main()