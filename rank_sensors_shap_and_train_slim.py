# rank_sensors_shap_and_train_slim.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import mlflow
import time
from pathlib import Path
from dataclasses import asdict, dataclass
import json
import h5py
import matplotlib.pyplot as plt
import shap  # Make sure you have shap installed: pip install shap

from libs.models import TemporalEncoder
from libs.data import get_prepared_data, loadData
from libs.test_encoder_predictor import find_model_paths, load_checkpoint as load_model_checkpoint
from parameters import Args as OriginalArgs
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})


# --- Configuration Dataclasses ---
@dataclass
class GlobalConfigSHAP:
    base_model_strdate_identifier: str = "20250906_19_38_58"  # <<< USER MUST SET THIS
    checkpoints_base_dir: str = "03_Checkpoints"
    results_base_dir: str = "04_Results"
    logs_base_dir: str = "02_Logs"
    n_test: int = 5000  # For creating dataloaders
    cuda: bool = True
    seed: int = 1234  # Different seed for this experiment
    torch_deterministic: bool = True
    comment: str = "SHAP sensor ranking and slim encoder training"


@dataclass
class SHAPConfig:
    run_phase: bool = True
    num_background_samples: int = 100  # For SHAP DeepExplainer background
    num_explanation_samples: int = 500  # Samples to explain to get SHAP values
    top_n_sensors_to_select: int = 4  # How many top sensors to pick for the slim encoder
    enforce_symmetry: bool = False  # If True, averages SHAP values for symmetric pairs and selects pairs


@dataclass
class SlimEncoderTrainingConfigSHAP:  # Renamed to avoid conflict if in same file later
    run_phase: bool = False
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 256


# --- Instantiate Configurations ---
GLOBAL_CONFIG_SHAP = GlobalConfigSHAP()
SHAP_CONFIG = SHAPConfig()
SLIM_CONFIG_SHAP = SlimEncoderTrainingConfigSHAP()

from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec

def plot_shap_sensor_maps(probes, in_dim_orig, selected_indices_shap,
                          final_shap_values_for_ranking, output_dir, output_filename_base):
    if not (probes.shape[0] == in_dim_orig and probes.shape[1] >= 2):
        print("Probe dimensions mismatch. Skipping sensor map plotting.")
        return None

    try:
        # --- Create figure with GridSpec for the two main plots ---
        # This gives us a well-defined layout to start with.
        fig = plt.figure(figsize=(8, 3))
        gs = GridSpec(2, 1, height_ratios=[1, 1], figure=fig, hspace=0.1)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # --- Truck outline geometry ---
        rectwidth = 7.65
        rectheight = 1.0
        radius = 0.118

        arc_top_angle = np.linspace(np.pi / 2, np.pi, 20)
        arc_top_x = radius + radius * np.cos(arc_top_angle)
        arc_top_y = (rectheight / 2 - radius) + radius * np.sin(arc_top_angle)

        arc_bottom_angle = np.linspace(np.pi, 3 * np.pi / 2, 20)
        arc_bottom_x = radius + radius * np.cos(arc_bottom_angle)
        arc_bottom_y = (-rectheight / 2 + radius) + radius * np.sin(arc_bottom_angle)

        verts = [
            (rectwidth, rectheight / 2), (radius, rectheight / 2),
            *zip(arc_top_x, arc_top_y), (0, -rectheight / 2 + radius),
            *zip(arc_bottom_x, arc_bottom_y), (rectwidth, -rectheight / 2),
            (rectwidth, rectheight / 2),
        ]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
        truck_path = MplPath(verts, codes)

        # --- SHAP plot (ax1) ---
        truck_patch1 = PathPatch(truck_path, facecolor='#f0f0f0', edgecolor='black', lw=1.0, zorder=0)
        ax1.add_patch(truck_patch1)
        sc = ax1.scatter(probes[:, 0], probes[:, 1], s=20,
                         c=final_shap_values_for_ranking, cmap='jet',
                         edgecolors='k', linewidths=0.25, zorder=10)

        # --- Robust Internal Colorbar using fig.add_axes ---
        # 1. Get the position of the main plot axis [left, bottom, width, height]
        #    in figure coordinates (fractions of the figure).
        ax1_pos = ax1.get_position()

        # 2. Define the position of the new colorbar axis (cax) relative to ax1_pos.
        #    This makes it robust to changes in figure size or layout.
        cax_width = ax1_pos.width * 0.60  # Colorbar width is 60% of the main plot's width
        cax_height = ax1_pos.height * 0.10  # Colorbar height is 15% of the main plot's height
        cax_left = ax1_pos.x0 + (ax1_pos.width - cax_width) / 2  # Center horizontally
        cax_bottom = ax1_pos.y1 - cax_height - (ax1_pos.height * 0.25)  # Position 10% from the top

        # 3. Add the new axis to the figure at the calculated position.
        cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])

        cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label('SHAP value')
        # Make the colorbar background transparent and remove the outline
        cbar.ax.patch.set_alpha(0.0)
        cbar.outline.set_visible(False)

        # Label (a)
        ax1.text(-0.4, rectheight / 2 + 0.2, '(a)')

        # --- Selected Sensors plot (ax2) ---
        truck_patch2 = PathPatch(truck_path, facecolor='#f0f0f0', edgecolor='black', lw=1.0, zorder=0)
        ax2.add_patch(truck_patch2)

        ax2.scatter(probes[:, 0], probes[:, 1], s=15, c='silver', edgecolors='grey',
                    linewidths=0.5, label='All', zorder=5)
        ax2.scatter(probes[selected_indices_shap, 0], probes[selected_indices_shap, 1],
                    s=25, c='#d62728', edgecolors='k', linewidths=0.5,
                    label='Selected', zorder=10)

        # Add internal legend
        ax2.legend(loc='center', ncol=2)

        # Label (b)
        ax2.text(-0.4, rectheight / 2 + 0.2, '(b)')

        # --- Final axis cleanup ---
        for ax in [ax1, ax2]:
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-0.1, rectwidth + 0.1)
            ax.set_ylim(-rectheight / 2 - 0.1, rectheight / 2 + 0.1)
            ax.axis('off')

        # --- Save in multiple formats ---
        combined_plot_path = Path(output_dir) / f"{output_filename_base}_shap_sensor_plots.png"
        # Using bbox_inches='tight' is crucial for removing excess whitespace and often works
        # better than tight_layout() when using manually placed axes.
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(combined_plot_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.savefig(combined_plot_path.with_suffix('.eps'), bbox_inches='tight')
        plt.close(fig)

        print(f"Saved combined SHAP sensor plots to {combined_plot_path}")
        return combined_plot_path

    except Exception as e:
        print(f"Error during SHAP sensor map plotting: {e}")
        return None


# --- Helper Function for Slim Encoder (can be reused) ---
def train_slim_encoder_shap(g_cfg: GlobalConfigSHAP, slim_cfg: SlimEncoderTrainingConfigSHAP,
                            original_args: OriginalArgs, device: torch.device,
                            dataloader_train: torch.utils.data.DataLoader,
                            dataloader_test: torch.utils.data.DataLoader,
                            encoder_orig_for_target: TemporalEncoder,  # Original encoder to produce target latents
                            selected_indices: np.ndarray,
                            slim_model_name: str,
                            slim_ckpdir: Path):
    if selected_indices is None or len(selected_indices) == 0:
        print("No sensors selected or selected_indices not provided. Skipping slim encoder training.")
        return None

    print("\n--- Initializing Slim Encoder (SHAP based) ---")
    slim_input_dim = len(selected_indices)
    print(f"Slim Encoder input_dim: {slim_input_dim}")

    encoder_slim = TemporalEncoder(
        input_dim=slim_input_dim, latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim, num_layers=original_args.n_layers,
        dropout_rate=original_args.dropout
    ).to(device)
    print("Initialized Slim Encoder (LSTM from scratch).")

    optimizer_slim = optim.Adam(encoder_slim.parameters(), lr=slim_cfg.lr)
    best_val_loss_slim = float('inf')
    best_slim_encoder_path = None

    for epoch in range(1, slim_cfg.epochs + 1):
        encoder_slim.train()
        epoch_slim_loss = 0
        for s_t_full, _, _, _ in dataloader_train:
            s_t_full = s_t_full.to(device)
            # Important: Select the features (sensors) for the slim encoder input
            s_t_for_slim_input = s_t_full[:, :, selected_indices]

            optimizer_slim.zero_grad()
            with torch.no_grad():
                # Target latents are from the original encoder using full sensor input
                z_t_target = encoder_orig_for_target(s_t_full)

            z_t_pred_slim = encoder_slim(s_t_for_slim_input)
            loss = F.mse_loss(z_t_pred_slim, z_t_target)
            loss.backward()
            optimizer_slim.step()
            epoch_slim_loss += loss.item()

        avg_epoch_slim_loss = epoch_slim_loss / len(dataloader_train)
        mlflow.log_metric("slim_train_latent_match_loss", avg_epoch_slim_loss, step=epoch)

        encoder_slim.eval()
        val_slim_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for s_t_full_val, _, _, _ in dataloader_test:
                s_t_full_val = s_t_full_val.to(device)
                s_t_slim_val_input = s_t_full_val[:, :, selected_indices]

                z_t_target_val = encoder_orig_for_target(s_t_full_val)
                z_t_pred_slim_val = encoder_slim(s_t_slim_val_input)
                val_slim_loss += F.mse_loss(z_t_pred_slim_val, z_t_target_val).item()
                num_val_batches += 1

        avg_val_slim_loss = val_slim_loss / num_val_batches
        mlflow.log_metric("slim_val_latent_match_loss", avg_val_slim_loss, step=epoch)
        print(
            f"Epoch {epoch}/{slim_cfg.epochs} [Slim-SHAP] - Train Loss: {avg_epoch_slim_loss:.6f} - Val Loss: {avg_val_slim_loss:.6f}")

        if avg_val_slim_loss < best_val_loss_slim:
            best_val_loss_slim = avg_val_slim_loss
            best_slim_encoder_path = slim_ckpdir / f"{slim_model_name}_slim_encoder.pth.tar"
            torch.save({'state_dict': encoder_slim.state_dict()}, best_slim_encoder_path)
            mlflow.log_artifact(str(best_slim_encoder_path), artifact_path="checkpoints_slim_shap")
            print(f"Saved best SHAP-based slim encoder: {best_slim_encoder_path}")
    return best_slim_encoder_path


def main_shap_workflow():
    if not GLOBAL_CONFIG_SHAP.base_model_strdate_identifier or GLOBAL_CONFIG_SHAP.base_model_strdate_identifier == "YYYYMMDD_HH_MM_SS":
        print("CRITICAL: `GLOBAL_CONFIG_SHAP.base_model_strdate_identifier` is not set.")
        return

    np.random.seed(GLOBAL_CONFIG_SHAP.seed)
    torch.manual_seed(GLOBAL_CONFIG_SHAP.seed)
    torch.backends.cudnn.deterministic = GLOBAL_CONFIG_SHAP.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and GLOBAL_CONFIG_SHAP.cuda else 'cpu')
    print(f"Using device: {device}")

    # --- Load Original Model Config and Paths ---
    print(f"Loading config for base model ID: {GLOBAL_CONFIG_SHAP.base_model_strdate_identifier}")
    base_case_ckp_dir, base_case_results_dir = find_model_paths(
        GLOBAL_CONFIG_SHAP.base_model_strdate_identifier, GLOBAL_CONFIG_SHAP.checkpoints_base_dir
    )
    if base_case_ckp_dir is None or base_case_results_dir is None: return

    latent_file_pattern = f"{GLOBAL_CONFIG_SHAP.base_model_strdate_identifier}*latent_space.hdf5"
    latent_files = sorted(list(base_case_results_dir.glob(latent_file_pattern)))
    base_latent_file = next((f for f in latent_files if '_eval' not in f.name),
                            latent_files[0] if latent_files else None)
    if not base_latent_file: print(f"Error: No base latent file. Exiting."); return

    with h5py.File(base_latent_file, 'r') as f:
        original_args_dict_loaded = json.loads(f.attrs['args'])
        original_args = OriginalArgs(**original_args_dict_loaded)

    # --- MLflow Setup ---
    strdate_main = time.strftime("%Y%m%d_%H_%M_%S")
    # Create a descriptive suffix for the run name based on the selection strategy
    run_name_suffix = f"{SHAP_CONFIG.top_n_sensors_to_select}sensors"
    run_name_suffix += "_sym" if SHAP_CONFIG.enforce_symmetry else "_asym"
    main_mlflow_run_name = f"{GLOBAL_CONFIG_SHAP.base_model_strdate_identifier}_{run_name_suffix}"

    case_mlflow_log_root = Path(GLOBAL_CONFIG_SHAP.logs_base_dir) / original_args.case
    case_mlflow_log_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{case_mlflow_log_root.resolve()}")
    mlflow_experiment_name_to_set = "sensor_selection_shap"  # Experiment name
    mlflow.set_experiment(mlflow_experiment_name_to_set)
    active_experiment = mlflow.get_experiment_by_name(mlflow_experiment_name_to_set)
    print(f"MLflow Experiment: '{active_experiment.name}', Run: '{main_mlflow_run_name}'")

    # --- Simplified Artifact Dirs (one folder per base model) ---
    base_artifact_dir_name = GLOBAL_CONFIG_SHAP.base_model_strdate_identifier
    current_run_ckp_dir = Path(GLOBAL_CONFIG_SHAP.checkpoints_base_dir) / original_args.case / "sensor_selection" / base_artifact_dir_name
    current_run_out_dir = Path(GLOBAL_CONFIG_SHAP.results_base_dir) / original_args.case / "sensor_selection" / base_artifact_dir_name
    current_run_ckp_dir.mkdir(parents=True, exist_ok=True)
    current_run_out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=main_mlflow_run_name, experiment_id=active_experiment.experiment_id) as run:
        mlflow.log_params(asdict(GLOBAL_CONFIG_SHAP))
        mlflow.log_params({"shapcfg_" + k: v for k, v in asdict(SHAP_CONFIG).items()})
        mlflow.log_params({"slimcfg_" + k: v for k, v in asdict(SLIM_CONFIG_SHAP).items()})
        if GLOBAL_CONFIG_SHAP.comment: mlflow.set_tag("run_comment", GLOBAL_CONFIG_SHAP.comment)

        # --- Load Data (a subset for SHAP, full for slim training) ---
        loader_args_shap = OriginalArgs(**original_args_dict_loaded)  # For SHAP data prep
        loader_args_shap.batch_size = max(SHAP_CONFIG.num_background_samples, SHAP_CONFIG.num_explanation_samples)
        loader_args_shap.n_test = 0  # Use training data for SHAP
        loader_args_shap.DATA_TO_GPU = False
        dataloader_for_shap, _ = get_prepared_data(loader_args_shap, device, shuffle_train=True,
                                                   shuffle_test=False)  # Shuffle to get diverse samples

        # --- Load PRE-TRAINED ORIGINAL ENCODER ---
        sample_s_t_full = dataloader_for_shap.dataset[0][0]
        in_dim_orig = sample_s_t_full.shape[-1]

        encoder_orig = TemporalEncoder(
            input_dim=in_dim_orig, latent_dim=original_args.latent_dim,
            hidden_dim=original_args.enc_hidden_dim, num_layers=original_args.n_layers,
            dropout_rate=original_args.dropout
        ).to(device)
        enc_path = base_case_ckp_dir / f"{original_args.modelname}_encoder.pth.tar"
        load_model_checkpoint(encoder_orig, str(enc_path), device)
        encoder_orig.eval()  # Ensure it's in eval mode for SHAP
        print(f"Loaded original encoder from {enc_path}")

        selected_indices_shap = None
        if SHAP_CONFIG.run_phase:
            print("\n--- Phase 1: SHAP Value Calculation and Sensor Ranking ---")

            shap_values_path = current_run_out_dir / "mean_abs_shap_values.txt"
            if shap_values_path.exists():
                print(f"Loading cached SHAP values from {shap_values_path}")
                mean_abs_shap_per_sensor = np.loadtxt(shap_values_path)
            else:
                print("Calculating SHAP values (will be cached for next run)...")
                # Prepare data for SHAP
                background_data_list = []
                for s_t, _, _, _ in dataloader_for_shap:
                    background_data_list.append(s_t)
                    if sum(b.shape[0] for b in background_data_list) >= SHAP_CONFIG.num_background_samples:
                        break
                background_s_t = torch.cat(background_data_list, dim=0)[:SHAP_CONFIG.num_background_samples].to(device)

                # Explanation data:
                start_idx_expl = SHAP_CONFIG.num_background_samples % len(dataloader_for_shap.dataset)
                temp_list_for_expl = []
                current_expl_count = 0
                for i, (s_t, _, _, _) in enumerate(dataloader_for_shap):
                    if i * dataloader_for_shap.batch_size < start_idx_expl: continue
                    temp_list_for_expl.append(s_t)
                    current_expl_count += s_t.shape[0]
                    if current_expl_count >= SHAP_CONFIG.num_explanation_samples:
                        break
                if not temp_list_for_expl:
                    explanation_s_t = background_s_t[:min(SHAP_CONFIG.num_explanation_samples, background_s_t.shape[0])].clone().to(device)
                else:
                    explanation_s_t = torch.cat(temp_list_for_expl, dim=0)[:SHAP_CONFIG.num_explanation_samples].to(device)

                print(f"Background data shape for SHAP: {background_s_t.shape}")
                print(f"Explanation data shape for SHAP: {explanation_s_t.shape}")

                explainer = shap.GradientExplainer(encoder_orig, background_s_t)

                # WORKAROUND for cuDNN RNN issue: Temporarily set to train mode for SHAP
                encoder_orig.train()
                shap_values_output = explainer.shap_values(explanation_s_t)
                encoder_orig.eval()

                # --- Start of aggregation ---
                if isinstance(shap_values_output, list):
                    # This is the standard expected output for multi-output models
                    # len(shap_values_output) == D (latent_dim)
                    # Each element shap_values_output[d] has shape (N, L, F)
                    print(f"SHAP output is a list of {len(shap_values_output)} arrays.")
                    if len(shap_values_output) > 0:
                        print(f"Shape of one element in SHAP output list: {shap_values_output[0].shape}")

                    stacked_abs_shap_values = np.array([np.abs(s_val) for s_val in shap_values_output])
                    summed_over_latents = np.sum(stacked_abs_shap_values, axis=0)
                    mean_abs_shap_per_sensor = np.mean(summed_over_latents, axis=(0, 1))

                elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 4:
                    # N = num_explanation_samples, L = lookback, F = features, D = latent_dim
                    print(f"SHAP output is a single 4D ndarray with shape: {shap_values_output.shape}")
                    abs_shap_values = np.abs(shap_values_output)
                    summed_over_latents = np.sum(abs_shap_values, axis=3)  # Shape: (N, L, F)
                    mean_abs_shap_per_sensor = np.mean(summed_over_latents, axis=(0, 1))  # Shape: (F,)

                elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 3:
                    # N = num_explanation_samples, L = lookback, F = features (for scalar output)
                    print(f"SHAP output is a single 3D ndarray with shape: {shap_values_output.shape}")
                    abs_shap_values = np.abs(shap_values_output)
                    mean_abs_shap_per_sensor = np.mean(abs_shap_values, axis=(0, 1))  # Shape: (F,)
                else:
                    raise TypeError(f"Unexpected type or ndim for shap_values_output: {type(shap_values_output)}, ndim: {getattr(shap_values_output, 'ndim', 'N/A')}")

                np.savetxt(shap_values_path, mean_abs_shap_per_sensor, fmt="%.6f")
                mlflow.log_artifact(str(shap_values_path), "shap_info")
                print(f"Saved calculated SHAP values to {shap_values_path}")

            print(f"Shape of mean_abs_shap_per_sensor: {mean_abs_shap_per_sensor.shape}")

            if SHAP_CONFIG.enforce_symmetry:
                print("\nSymmetrizing SHAP scores...")
                try:
                    with open('sensor_symmetry_map.json', 'r') as f:
                        map_dict = json.load(f)
                except FileNotFoundError:
                    print("Error: sensor_symmetry_map.json not found. Cannot enforce symmetry.")
                    return

                num_sensors_total = in_dim_orig
                sensor_map = list(range(num_sensors_total))
                for k, v in map_dict.items():
                    sensor_map[int(k)] = v

                symmetrized_shap_values = mean_abs_shap_per_sensor.copy()
                processed_for_symm = set()
                for i in range(num_sensors_total):
                    if i in processed_for_symm: continue
                    partner_idx = sensor_map[i]
                    avg_shap = (mean_abs_shap_per_sensor[i] + mean_abs_shap_per_sensor[partner_idx]) / 2.0
                    symmetrized_shap_values[i] = avg_shap
                    symmetrized_shap_values[partner_idx] = avg_shap
                    processed_for_symm.add(i); processed_for_symm.add(partner_idx)
                print("SHAP scores have been symmetrized by averaging pairs.")
                final_shap_values_for_ranking = symmetrized_shap_values

                # --- Select top N/2 PAIRS based on symmetrized scores ---
                sensor_ranking_indices = np.argsort(final_shap_values_for_ranking)[::-1]
                selected_pairs, processed_indices = [], set()
                for sensor_idx in sensor_ranking_indices:
                    if sensor_idx in processed_indices: continue
                    partner_idx = sensor_map[sensor_idx]
                    selected_pairs.append(tuple(sorted((sensor_idx, partner_idx))))
                    processed_indices.add(sensor_idx); processed_indices.add(partner_idx)
                    if len(selected_pairs) >= SHAP_CONFIG.top_n_sensors_to_select // 2: break
                _selected_indices_shap = np.array(sorted([idx for pair in selected_pairs for idx in pair]))

            else: # Asymmetric selection
                print("\nUsing raw SHAP scores for ranking (symmetry not enforced).")
                final_shap_values_for_ranking = mean_abs_shap_per_sensor
                sensor_ranking_indices = np.argsort(final_shap_values_for_ranking)[::-1]
                _selected_indices_shap = sensor_ranking_indices[:SHAP_CONFIG.top_n_sensors_to_select]
                _selected_indices_shap.sort()

            selected_indices_shap = _selected_indices_shap.copy()  # Make contiguous copy

            print(
                f"Top {SHAP_CONFIG.top_n_sensors_to_select} sensor indices from SHAP (contiguous copy): {selected_indices_shap.tolist()}")

            # Log SHAP results
            np.savetxt(current_run_out_dir / f"{main_mlflow_run_name}_shap_mean_abs_values.txt",
                       mean_abs_shap_per_sensor, fmt="%.6f")
            mlflow.log_artifact(str(current_run_out_dir / f"{main_mlflow_run_name}_shap_mean_abs_values.txt"),
                                "shap_info")
            np.savetxt(current_run_out_dir / f"{main_mlflow_run_name}_shap_selected_indices.txt", selected_indices_shap,
                       fmt="%d")
            mlflow.log_artifact(str(current_run_out_dir / f"{main_mlflow_run_name}_shap_selected_indices.txt"),
                                "shap_info")
            mlflow.log_metric("shap_num_selected_sensors", len(selected_indices_shap))

            # --- Plotting SHAP Importances ---
            plt.figure(figsize=(12, 7))
            plt.bar(range(len(final_shap_values_for_ranking)), final_shap_values_for_ranking, color='skyblue')
            plt.xlabel("Sensor Index")
            plot_ylabel = "Symmetrized Mean Abs SHAP" if SHAP_CONFIG.enforce_symmetry else "Mean Abs SHAP"
            plt.ylabel(plot_ylabel)
            plt.title(f"SHAP Sensor Importance ({run_name_suffix})\nBase Model: {base_artifact_dir_name}")
            plt.tight_layout()
            shap_plot_path = current_run_out_dir / f"{main_mlflow_run_name}_shap_importances.png"
            plt.savefig(shap_plot_path)
            plt.savefig(shap_plot_path.with_suffix('.pdf'))
            plt.savefig(shap_plot_path.with_suffix('.eps'))
            plt.close()
            mlflow.log_artifact(str(shap_plot_path), "shap_plots")

            # --- Combined Sensor Location and Importance Plot ---
            try:
                (_, _, _, _, _, _, _, _, _, probes, _) = loadData(original_args, sel_coefs=['Cd', 'Cl'],
                                                                  printer=False)
                # New function call replaces the old plotting code
                combined_plot_path = plot_shap_sensor_maps(
                    probes=probes,
                    in_dim_orig=in_dim_orig,
                    selected_indices_shap=selected_indices_shap,
                    final_shap_values_for_ranking=final_shap_values_for_ranking,
                    output_dir=current_run_out_dir,
                    output_filename_base=main_mlflow_run_name
                )
                if combined_plot_path:
                    # Log the combined plot and its vector formats
                    mlflow.log_artifact(str(combined_plot_path), "shap_plots")
                    mlflow.log_artifact(str(Path(combined_plot_path).with_suffix('.pdf')), "shap_plots")
                    mlflow.log_artifact(str(Path(combined_plot_path).with_suffix('.eps')), "shap_plots")

            except Exception as e:
                print(f"Error loading probe data for plotting: {e}")


        else:
            print("\n--- Skipping Phase 1: SHAP Sensor Ranking ---")
            # If skipping, try to load pre-computed indices if slim training is enabled
            if SLIM_CONFIG_SHAP.run_phase:
                try:
                    # You'd need a mechanism to specify which pre-computed file to load.
                    # For now, this is a placeholder.
                    # selected_indices_shap = np.loadtxt("path_to_precomputed_shap_indices.txt", dtype=int)
                    print(
                        "SHAP phase skipped. Slim encoder training requires selected_indices_shap to be loaded or computed.")
                    # return # Or handle appropriately
                except FileNotFoundError:
                    print("SHAP phase skipped and no pre-computed indices found. Cannot train slim encoder.")
                    return

        if SLIM_CONFIG_SHAP.run_phase:
            if selected_indices_shap is None or len(selected_indices_shap) == 0:
                print(
                    "CRITICAL: Slim Encoder training (SHAP) enabled, but no sensors selected from SHAP Phase. Halting.")
                return

            # Prepare dataloaders for slim encoder training (might need different batch size)
            loader_args_slim = OriginalArgs(**original_args_dict_loaded)
            loader_args_slim.batch_size = SLIM_CONFIG_SHAP.batch_size
            loader_args_slim.DATA_TO_GPU = False
            loader_args_slim.n_test = GLOBAL_CONFIG_SHAP.n_test  # Use global n_test
            dataloader_train_slim, dataloader_test_slim = get_prepared_data(loader_args_slim, device,
                                                                            shuffle_train=True, shuffle_test=False)
            if not dataloader_train_slim.dataset or (
                    GLOBAL_CONFIG_SHAP.n_test > 0 and not dataloader_test_slim.dataset):
                print("Error: Slim Training or Test dataloader is empty.")
                return

            print("\n--- Phase 2: Training Slim Encoder (SHAP based) ---")
            slim_model_name = main_mlflow_run_name  # Use the descriptive run name for the model
            train_slim_encoder_shap(
                GLOBAL_CONFIG_SHAP, SLIM_CONFIG_SHAP, original_args, device,
                dataloader_train_slim, dataloader_test_slim,
                encoder_orig,  # Pass original encoder for generating target latents
                selected_indices_shap,
                slim_model_name, current_run_ckp_dir
            )
        else:
            print("\n--- Skipping Phase 2: Slim Encoder Training (SHAP based) ---")

        print(f"\nSHAP-based optimization finished. MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main_shap_workflow()