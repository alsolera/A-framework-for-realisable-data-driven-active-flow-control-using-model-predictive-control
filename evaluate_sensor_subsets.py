# evaluate_sensor_subsets.py
import matplotlib
matplotlib.use('Agg')  # Set the backend BEFORE importing pyplot
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass, field
import json
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import r2_score

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
    'savefig.dpi': 300
})


# --- Configuration ---
@dataclass
class EvalConfig:
    """Configuration for the sensor subset evaluation study."""
    base_model_strdate_identifier: str = "20250906_19_38_58"  # <<< SET THIS to the ID of the full model
    checkpoints_base_dir: str = "03_Checkpoints"
    results_base_dir: str = "04_Results"

    # Define the number of sensors for each slim encoder to be trained and evaluated
    sensor_subsets_to_eval: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 90])
    enforce_symmetry: bool = False  # Use symmetric pairing for sensor selection

    # Slim encoder training parameters
    slim_training_epochs: int = 100
    slim_training_lr: float = 1e-3
    slim_training_batch_size: int = 256

    # General settings
    n_test_for_eval: int = 5000  # Number of test samples for final force evaluation
    cuda: bool = True
    seed: int = 4321  # A different seed for this specific experiment
    torch_deterministic: bool = True


# --- Instantiate Configuration ---
CONFIG = EvalConfig()


def train_slim_encoder(
        slim_encoder: TemporalEncoder,
        encoder_orig_for_target: TemporalEncoder,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        selected_indices: np.ndarray,
        cfg: EvalConfig,
        device: torch.device,
        checkpoint_path: Path
):
    """Trains a slim encoder to match the latent space of a full encoder."""
    optimizer = optim.Adam(slim_encoder.parameters(), lr=cfg.slim_training_lr)
    best_val_loss = float('inf')

    pbar = tqdm(range(1, cfg.slim_training_epochs + 1), desc=f"Training {len(selected_indices)}-sensor encoder", leave=False)
    for epoch in pbar:
        slim_encoder.train()
        for s_t_full, _, _, _ in dataloader_train:
            s_t_full = s_t_full.to(device)
            s_t_slim_input = s_t_full[:, :, selected_indices]

            optimizer.zero_grad()
            with torch.no_grad():
                z_t_target = encoder_orig_for_target(s_t_full)
            z_t_pred_slim = slim_encoder(s_t_slim_input)

            loss = F.mse_loss(z_t_pred_slim, z_t_target)
            loss.backward()
            optimizer.step()

        # Validation
        slim_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for s_t_full_val, _, _, _ in dataloader_test:
                s_t_full_val = s_t_full_val.to(device)
                s_t_slim_val_input = s_t_full_val[:, :, selected_indices]
                z_t_target_val = encoder_orig_for_target(s_t_full_val)
                z_t_pred_slim_val = slim_encoder(s_t_slim_val_input)
                val_loss += F.mse_loss(z_t_pred_slim_val, z_t_target_val).item()

        avg_val_loss = val_loss / len(dataloader_test)
        pbar.set_postfix({'best_val_loss': f'{best_val_loss:.6f}'})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'state_dict': slim_encoder.state_dict()}, checkpoint_path)

    print(f"Finished training. Best validation loss: {best_val_loss:.6f}. Model saved to {checkpoint_path}")
    return checkpoint_path


def evaluate_force_error(
        encoder: TemporalEncoder,
        force_decoder: ForceDecoder,
        dataloader_test: torch.utils.data.DataLoader,
        raw_forces_ground_truth: np.ndarray,
        selected_indices: np.ndarray,
        device: torch.device,
        forces_mean_tensor: torch.Tensor,
        forces_std_tensor: torch.Tensor
):
    """
    Evaluates force prediction metrics (MAE, R2) by aligning predictions with the full raw dataset.
    This mirrors the logic in test_encoder_predictor.py for consistency.
    """
    encoder.eval()
    force_decoder.eval()

    # 1. Get all predictions from the encoder -> decoder pipeline
    y_pred_scaled_list = []
    # Use a simple dataloader on the dataset part of the provided loader
    temp_loader = DataLoader(dataloader_test.dataset, batch_size=512, shuffle=False)
    with torch.no_grad():
        for (s_t_full, _, _, _) in temp_loader:
            s_t_input = s_t_full[:, :, selected_indices].to(device)
            z_t = encoder(s_t_input)
            preds_scaled = force_decoder(z_t)
            y_pred_scaled_list.append(preds_scaled.cpu())

    y_pred_scaled = torch.cat(y_pred_scaled_list).numpy()

    # 2. Unscale predictions
    y_pred_unscaled = y_pred_scaled * forces_std_tensor.cpu().numpy() + forces_mean_tensor.cpu().numpy()

    # 3. Align true forces from the raw dataset
    lookback = dataloader_test.dataset.tensors[0].shape[1]
    force_offset = lookback - 1
    num_predictions = len(y_pred_unscaled)
    y_true_unscaled = raw_forces_ground_truth[force_offset : force_offset + num_predictions]

    if len(y_true_unscaled) != num_predictions:
        print(f"FATAL: Alignment mismatch. Have {num_predictions} predictions but can only align {len(y_true_unscaled)} true values.")
        raise ValueError("Ground truth and prediction lengths do not match after alignment.")

    # Calculate MAE for Cd and Cl
    mae_cd = np.mean(np.abs(y_pred_unscaled[:, 0] - y_true_unscaled[:, 0]))
    mae_cl = np.mean(np.abs(y_pred_unscaled[:, 1] - y_true_unscaled[:, 1]))

    # Calculate R2 score for Cd and Cl
    r2 = r2_score(y_true_unscaled, y_pred_unscaled, multioutput='raw_values')
    r2_cd, r2_cl = r2[0], r2[1]

    print(f"Force Prediction with {len(selected_indices)} sensors: MAE Cd={mae_cd:.6f}, Cl={mae_cl:.6f}, R2 Cd={r2_cd:.4f}, R2 Cl={r2_cl:.4f}")
    return mae_cd, mae_cl, r2_cd, r2_cl


def plot_results(results_data: list, output_path: Path, in_dim_orig: int):
    """Plots the relative force prediction MAE for Cd and Cl vs. the number of sensors."""
    if not results_data:
        print("No results to plot.")
        return

    # Sort results by number of sensors
    results_data.sort(key=lambda x: x['num_sensors'])

    # Find the baseline result for the full model
    baseline_result = next((r for r in results_data if r['num_sensors'] == in_dim_orig), None)

    # Separate plot data from the baseline
    plot_data = [r for r in results_data if r.get('num_sensors') != in_dim_orig]
    num_sensors = [r['num_sensors'] for r in plot_data]
    norm_mae_cd_errors = [r['norm_mae_cd'] for r in plot_data]
    norm_mae_cl_errors = [r['norm_mae_cl'] for r in plot_data]

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(num_sensors, norm_mae_cd_errors, 'o-', color='tab:blue', markerfacecolor='tab:blue', markersize=6, label='$C_d$')
    ax.plot(num_sensors, norm_mae_cl_errors, 's-', color='tab:orange', markerfacecolor='tab:orange', markersize=6, label='$C_l$')

    # Plot horizontal lines for the baseline
    if baseline_result:
        ax.axhline(y=baseline_result['norm_mae_cd'], color='tab:blue', linestyle=':', linewidth=1.5,
                   label=f'$C_d$ Full model')
        ax.axhline(y=baseline_result['norm_mae_cl'], color='tab:orange', linestyle=':', linewidth=1.5,
                   label=f'$C_l$ Full model')

    ax.set_xscale('log', base=2)
    ax.set_xticks(num_sensors)
    ax.set_xticklabels(num_sensors)

    ax.set_xlabel("Number of sensors (Log scale)")
    ax.set_ylabel(f"Relative error ($L_1/\sigma$)")
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.legend(ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.eps'))
    print(f"Saved results plot to {output_path}")
    plt.close(fig)


def plot_r2_results(results_data: list, output_path: Path, in_dim_orig: int):
    """Plots the R2 score for Cd and Cl vs. the number of sensors."""
    if not results_data:
        print("No results to plot for R2.")
        return

    # Sort results by number of sensors
    results_data.sort(key=lambda x: x['num_sensors'])

    # Find the baseline result for the full model
    baseline_result = next((r for r in results_data if r['num_sensors'] == in_dim_orig), None)

    # Separate plot data from the baseline
    plot_data = [r for r in results_data if r.get('num_sensors') != in_dim_orig]
    num_sensors = [r['num_sensors'] for r in plot_data]
    r2_cd_scores = [r['r2_cd'] for r in plot_data]
    r2_cl_scores = [r['r2_cl'] for r in plot_data]

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(num_sensors, r2_cd_scores, 'o-', color='tab:blue', markerfacecolor='tab:blue', markersize=6, label='$C_d$')
    ax.plot(num_sensors, r2_cl_scores, 's-', color='tab:orange', markerfacecolor='tab:orange', markersize=6, label='$C_l$')

    # Plot horizontal lines for the baseline
    if baseline_result:
        ax.axhline(y=baseline_result['r2_cd'], color='tab:blue', linestyle=':', linewidth=1.5, label=f'$C_d$ Full model')
        ax.axhline(y=baseline_result['r2_cl'], color='tab:orange', linestyle=':', linewidth=1.5, label=f'$C_l$ Full model')

    ax.set_xscale('log', base=2)
    ax.set_xticks(num_sensors)
    ax.set_xticklabels(num_sensors)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))  # Ensure y-axis starts at 0 or higher

    ax.set_xlabel("Number of sensors (Log scale)")
    ax.set_ylabel("$R^2$ Score")
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.legend(ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.eps'))
    print(f"Saved R2 results plot to {output_path}")
    plt.close(fig)


def plot_combined_results(results_data: list, output_path: Path, in_dim_orig: int):
    """Plots both relative MAE and R2 score vs. number of sensors side-by-side."""
    if not results_data:
        print("No results to plot.")
        return

    # Sort results by number of sensors
    results_data.sort(key=lambda x: x['num_sensors'])

    # Find the baseline result for the full model
    baseline_result = next((r for r in results_data if r['num_sensors'] == in_dim_orig), None)

    # Separate plot data from the baseline
    plot_data = [r for r in results_data if r.get('num_sensors') != in_dim_orig]
    num_sensors = [r['num_sensors'] for r in plot_data]

    # --- Create Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # --- Subplot 1: Relative MAE ---
    norm_mae_cd_errors = [r['norm_mae_cd'] for r in plot_data]
    norm_mae_cl_errors = [r['norm_mae_cl'] for r in plot_data]

    ax1.plot(num_sensors, norm_mae_cd_errors, 'o-', color='tab:blue', markerfacecolor='tab:blue', markersize=4, label='$C_d$')
    ax1.plot(num_sensors, norm_mae_cl_errors, 's-', color='tab:orange', markerfacecolor='tab:orange', markersize=4, label='$C_l$')
    if baseline_result:
        ax1.axhline(y=baseline_result['norm_mae_cd'], color='tab:blue', linestyle=':', linewidth=1.5, label=f'$C_d$ Full model')
        ax1.axhline(y=baseline_result['norm_mae_cl'], color='tab:orange', linestyle=':', linewidth=1.5, label=f'$C_l$ Full model')

    ax1.set_xscale('log', base=2); ax1.set_xticks(num_sensors); ax1.set_xticklabels(num_sensors)
    ax1.set_xlabel("Number of sensors (Log scale)"); ax1.set_ylabel(r"Relative error ($L_1/\sigma$)")
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, size='large')

    # --- Subplot 2: R2 Score ---
    r2_cd_scores = [r['r2_cd'] for r in plot_data]
    r2_cl_scores = [r['r2_cl'] for r in plot_data]

    ax2.plot(num_sensors, r2_cd_scores, 'o-', color='tab:blue', markerfacecolor='tab:blue', markersize=4, label='$C_d$')
    ax2.plot(num_sensors, r2_cl_scores, 's-', color='tab:orange', markerfacecolor='tab:orange', markersize=4, label='$C_l$')
    if baseline_result:
        ax2.axhline(y=baseline_result['r2_cd'], color='tab:blue', linestyle=':', linewidth=1.5, label=f'$C_d$ Full model')
        ax2.axhline(y=baseline_result['r2_cl'], color='tab:orange', linestyle=':', linewidth=1.5, label=f'$C_l$ Full model')

    ax2.set_xscale('log', base=2); ax2.set_xticks(num_sensors); ax2.set_xticklabels(num_sensors)
    ax2.set_ylim(bottom=max(0, ax2.get_ylim()[0])); ax2.set_xlabel("Number of sensors (Log scale)")
    ax2.set_ylabel("$R^2$")
    ax2.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, size='large')

    # --- Create a single, shared legend ---
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Create a unique set of handles and labels
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4,
               frameon=False)

    # --- Finalize and Save ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for the legend
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved combined results plot to {output_path}"); plt.close(fig)


def plot_selected_sensors(
    all_probes: np.ndarray,
    selected_indices: np.ndarray,
    output_path: Path
):
    """Plots the locations of all sensors and highlights the selected ones."""
    num_selected = len(selected_indices)
    num_total = len(all_probes)

    fig, ax = plt.subplots(figsize=(8, 3)) # Adjusted for better aspect ratio

    # Plot all sensors
    ax.scatter(all_probes[:, 0], all_probes[:, 1], s=8, c='lightgray', label=f'All Sensors ({num_total})')
    # Highlight selected sensors
    ax.scatter(all_probes[selected_indices, 0], all_probes[selected_indices, 1], s=15, c='red',
                label=f'Selected (N={num_selected})', zorder=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'SHAP Selected Sensor Locations (Top {num_selected})')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved selected sensor plot to {output_path}")


def load_raw_forces(args_for_data, sel_coefs=['Cd', 'Cl']):
    """Loads the full, raw unscaled force data from a dataset."""
    print("Loading raw unscaled forces for evaluation alignment...")
    try:
        class TempLoadArgs:
            datafile = args_for_data.datafile
            augment_with_symmetry = False # Never augment for evaluation
        (_, forces_unscaled_eval, _, _, _, _, _, _, _, _, _) = loadData(TempLoadArgs, sel_coefs=sel_coefs)
        print(f"Loaded raw evaluation forces. Shape: {forces_unscaled_eval.shape}")
        return forces_unscaled_eval
    except Exception as e:
        print(f"FATAL: Could not load forces from eval dataset for alignment: {e}")
        raise


def main():
    """Main workflow to evaluate sensor subsets."""
    # --- Basic Setup ---
    if not CONFIG.base_model_strdate_identifier or CONFIG.base_model_strdate_identifier == "YYYYMMDD_HH_MM_SS":
        print("CRITICAL: `CONFIG.base_model_strdate_identifier` is not set.")
        return
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.backends.cudnn.deterministic = CONFIG.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and CONFIG.cuda else 'cpu')
    print(f"Using device: {device}")

    # --- Load Original Model Config and Paths ---
    print(f"Loading config for base model ID: {CONFIG.base_model_strdate_identifier}")
    base_case_ckp_dir, base_case_results_dir = find_model_paths(
        CONFIG.base_model_strdate_identifier, CONFIG.checkpoints_base_dir
    )
    if base_case_ckp_dir is None or base_case_results_dir is None: return

    latent_file_pattern = f"{CONFIG.base_model_strdate_identifier}*latent_space.hdf5"
    base_latent_file = next(
        (f for f in sorted(list(base_case_results_dir.glob(latent_file_pattern))) if '_eval' not in f.name), None)
    if not base_latent_file: print(f"Error: No base latent file found. Exiting."); return

    with h5py.File(base_latent_file, 'r') as f:
        original_args_dict = json.loads(f.attrs['args'])
        original_args = OriginalArgs(**original_args_dict)
        # Load scaling factors for unscaling the force predictions
        forces_mean = f['forces_mean'][:]
        forces_std = f['forces_std'][:]

    # Create tensors for scaling factors
    forces_mean_tensor = torch.tensor(forces_mean, dtype=torch.float32, device=device)
    forces_std_tensor = torch.tensor(forces_std, dtype=torch.float32, device=device)

    # --- Create dedicated output directory for this evaluation ---
    base_artifact_dir = Path(
        CONFIG.results_base_dir) / original_args.case / "sensor_selection" / CONFIG.base_model_strdate_identifier
    output_dir = base_artifact_dir / "subset_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts for this run will be saved in: {output_dir}")

    # --- Load Cached SHAP Values ---
    shap_values_path = base_artifact_dir / "mean_abs_shap_values.txt"
    if not shap_values_path.exists():
        print(f"Error: SHAP values not found at {shap_values_path}.")
        print("Please run `rank_sensors_shap_and_train_slim.py` first to generate and cache the SHAP values.")
        return
    mean_abs_shap_per_sensor = np.loadtxt(shap_values_path)
    print(f"Successfully loaded cached SHAP values from {shap_values_path}")

    # --- Load Probe locations for plotting ---
    try:
        # We only need the probe locations from loadData
        (_, _, _, _, _, _, _, _, _, probes, _) = loadData(original_args, sel_coefs=['Cd', 'Cl'], printer=False)
        print(f"Loaded {probes.shape[0]} probe locations for plotting.")
    except Exception as e:
        print(f"Warning: Could not load probe locations for plotting. Error: {e}")
        probes = None

    # --- Load Pre-trained Full Encoder and Force Decoder ---
    in_dim_orig = mean_abs_shap_per_sensor.shape[0]
    encoder_orig = TemporalEncoder(input_dim=in_dim_orig, latent_dim=original_args.latent_dim,
                                   hidden_dim=original_args.enc_hidden_dim, num_layers=original_args.n_layers).to(
        device)
    force_decoder = ForceDecoder(latent_dim=original_args.latent_dim, hidden_dim=original_args.forces_hidden_dim,
                                 arch=original_args.force_decoder_arch).to(device)

    enc_path = base_case_ckp_dir / f"{original_args.modelname}_encoder.pth.tar"
    force_path = base_case_ckp_dir / f"{original_args.modelname}_force.pth.tar"
    load_checkpoint(encoder_orig, str(enc_path), device)
    load_checkpoint(force_decoder, str(force_path), device)
    encoder_orig.eval();
    force_decoder.eval()
    print("Loaded original full encoder and force decoder.")

    # --- Prepare Dataloaders ---
    # Dataloader for slim encoder training
    loader_args_train = deepcopy(original_args)
    loader_args_train.batch_size = CONFIG.slim_training_batch_size
    loader_args_train.n_test = CONFIG.n_test_for_eval  # Use same test split
    dl_train_slim, dl_test_slim = get_prepared_data(loader_args_train, device, shuffle_train=True, shuffle_test=False)

    # Dataloader for final force evaluation (on test set, no shuffle)
    # CRITICAL: This loader must contain the ENTIRE evaluation dataset for correct alignment.
    loader_args_eval = deepcopy(original_args)
    project_root = Path(CONFIG.checkpoints_base_dir).parent
    eval_dataset_paths_raw = original_args.eval_dataset
    if isinstance(eval_dataset_paths_raw, str):
        eval_dataset_paths_raw = [eval_dataset_paths_raw]
    loader_args_eval.datafile = [str(project_root / p) for p in eval_dataset_paths_raw]
    loader_args_eval.batch_size = 512  # Can use a larger batch size for eval
    loader_args_eval.n_test = 0 # Use all data from the eval file
    loader_args_eval.augment_with_symmetry = False # NEVER augment evaluation data
    # By setting n_test=0, the 'train' loader will contain the full dataset. We use this for evaluation.
    dl_force_eval, _ = get_prepared_data(loader_args_eval, device, shuffle_train=False, shuffle_test=False)

    # --- Load the full raw forces for correct ground truth alignment ---
    # The dataloader returns future forces; we need the force at time t.
    raw_forces_unscaled_eval = load_raw_forces(loader_args_eval)

    # --- Calculate Std. Dev. of true forces on the test set for normalization ---
    print("Calculating standard deviation of forces on the test set...")
    # Use the std of the raw loaded forces for the most accurate normalization
    forces_std_test_set = np.std(raw_forces_unscaled_eval, axis=0)
    print(f"Test set Std. Dev.: Cd={forces_std_test_set[0]:.4f}, Cl={forces_std_test_set[1]:.4f}")


    # --- Main Loop ---
    results_file = output_dir / "sensor_subset_errors.json"
    results = []
    final_shap_values_for_ranking = mean_abs_shap_per_sensor

    if CONFIG.enforce_symmetry:
        print("\nSymmetrizing SHAP scores for ranking...")
        try:
            with open('sensor_symmetry_map.json', 'r') as f:
                map_dict = json.load(f)
            sensor_map = list(range(in_dim_orig))
            for k, v in map_dict.items():
                sensor_map[int(k)] = v
        except FileNotFoundError:
            print("Error: sensor_symmetry_map.json not found."); return

        symmetrized_values = np.copy(mean_abs_shap_per_sensor)
        processed = set()
        for i in range(in_dim_orig):
            if i in processed: continue
            partner_idx = sensor_map[i]
            avg_shap = (symmetrized_values[i] + symmetrized_values[partner_idx]) / 2.0
            symmetrized_values[i] = symmetrized_values[partner_idx] = avg_shap
            processed.add(i);
            processed.add(partner_idx)
        final_shap_values_for_ranking = symmetrized_values

    sensor_ranking_indices = np.argsort(final_shap_values_for_ranking)[::-1]

    for num_sensors in tqdm(sorted(CONFIG.sensor_subsets_to_eval), desc="Processing sensor subsets"):
        # 1. Select Sensors
        if num_sensors == 1:
            selected_indices = np.array([sensor_ranking_indices[0]])
        elif CONFIG.enforce_symmetry:
            # Skip odd numbers > 1 for symmetric selection
            if num_sensors % 2 != 0:
                print(f"Skipping odd number of sensors ({num_sensors}) for symmetric selection.")
                continue
            selected_pairs, processed_indices = [], set()
            for sensor_idx in sensor_ranking_indices:
                if sensor_idx in processed_indices: continue
                partner_idx = sensor_map[sensor_idx]
                selected_pairs.append(tuple(sorted((sensor_idx, partner_idx))))
                processed_indices.add(sensor_idx);
                processed_indices.add(partner_idx)
                if len(selected_pairs) >= num_sensors // 2: break
            selected_indices = np.array(sorted([idx for pair in selected_pairs for idx in pair]))
        else:
            selected_indices = np.sort(sensor_ranking_indices[:num_sensors])

        # Plot the selected sensor locations for this subset
        if probes is not None:
            plot_filename = output_dir / f"selected_locations_{num_sensors}sensors.png"
            plot_selected_sensors(probes, selected_indices, plot_filename)

        # Handle the full sensor set as a special baseline case (no training)
        if num_sensors >= in_dim_orig:
            print(f"\nEvaluating baseline with full {in_dim_orig} sensors...")
            eval_encoder = encoder_orig
            selected_indices = np.arange(in_dim_orig) # Use all sensors
        else:
            # 2. Train a new Slim Encoder
            ckpt_path = output_dir / f"slim_encoder_{num_sensors}sensors.pth.tar"
            slim_encoder = TemporalEncoder(input_dim=len(selected_indices), latent_dim=original_args.latent_dim,
                                           hidden_dim=original_args.enc_hidden_dim,
                                           num_layers=original_args.n_layers).to(device)

            if ckpt_path.exists():
                print(f"\nFound existing checkpoint for {num_sensors} sensors. Skipping training.")
            else:
                train_slim_encoder(slim_encoder, encoder_orig, dl_train_slim, dl_test_slim, selected_indices, CONFIG, device, ckpt_path)

            load_checkpoint(slim_encoder, ckpt_path, device)  # Load best saved model for evaluation
            eval_encoder = slim_encoder

        # 3. Evaluate Force Prediction Error using the appropriate encoder
        # We now pass the separately loaded raw forces as the ground truth.
        mae_cd, mae_cl, r2_cd, r2_cl = evaluate_force_error(
            eval_encoder, force_decoder, dl_force_eval, raw_forces_unscaled_eval,
            selected_indices, device, forces_mean_tensor, forces_std_tensor
        )
        # 4. Store Result (normalized)
        norm_mae_cd = mae_cd / forces_std_test_set[0]
        norm_mae_cl = mae_cl / forces_std_test_set[1]
        results.append({
            'num_sensors': num_sensors,
            'norm_mae_cd': float(norm_mae_cd),
            'norm_mae_cl': float(norm_mae_cl),
            'r2_cd': float(r2_cd),
            'r2_cl': float(r2_cl)
        })

    # Save results to JSON file after loop finishes
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved all evaluation results to {results_file}")

    # --- Final Plotting ---
    plot_path = output_dir / "error_vs_sensors.png"
    plot_results(results, plot_path, in_dim_orig)

    r2_plot_path = output_dir / "r2_vs_sensors.png"
    plot_r2_results(results, r2_plot_path, in_dim_orig)

    combined_plot_path = output_dir / "combined_metrics_vs_sensors.png"
    plot_combined_results(results, combined_plot_path, in_dim_orig)

    print("\nEvaluation script finished.")


if __name__ == "__main__":
    main()