import h5py
from datetime import datetime
from gymprecice.utils.fileutils import make_result_dir
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For initialization progress
import time
from libs.env_wrappers import DelayObservationWrapper, CustomRescaleAction
from libs.MPC import MPCControllerLatent
import gymnasium as gym
import torch
from pathlib import Path


# Modified get_prev_obs to handle potential errors gracefully
def get_prev_obs(file_path, lookback):
    """Load previous observation history (sensors only)."""
    file_path = Path(file_path)  # Ensure it's a Path object
    if not file_path.is_file():
        raise FileNotFoundError(f"Initial history file not found: {file_path}")

    try:
        with h5py.File(file_path, 'r') as f:
            if 'data' not in f:
                raise KeyError("'data' key not found in HDF5 file.")
            # Assuming data is stored under 'data' key and contains ONLY sensor values
            obs = f['data'][:]
        print(f"Loaded previous observations from {file_path}, shape: {obs.shape}")

        if obs.shape[0] < lookback:
            raise ValueError(f"File {file_path} has only {obs.shape[0]} steps, less than required lookback {lookback}")

        # Return only the required lookback portion
        obs_history = obs[-lookback:]
        print(f"Using last {lookback} steps for initial history.")
        return obs_history
    except Exception as e:
        print(f"Error loading initial observations from {file_path}: {e}")
        raise  # Re-raise the exception


# Modified get_env to not take history and remove sensor selection
def get_env(environment_config, lookback):
    """Initializes and wraps the Gym environment."""
    try:
        # Make sure environment.py is correctly located relative to this script
        # Example: from simulation.environment import JetTruck2DEnv
        # Adjust the import path as needed based on your project structure
        from libs.environment import JetTruck2DEnv  # Use the local environment definition
    except ImportError:
        print("Error: Could not import JetTruck2DEnv from libs.environment.")  # Adjusted path
        print("Ensure libs/environment.py is in the correct location or added to PYTHONPATH.")
        raise

    print("Initializing base environment...")
    base_env = JetTruck2DEnv(environment_config, 0, openloop=False)
    assert isinstance(base_env.action_space, gym.spaces.Box), "Only continuous action space is supported"

    # Configure environment timings (ensure consistency or make configurable)
    delta_t_cfd = 0.001
    delta_t_control = 0.2
    base_env.action_interval = int(round(delta_t_control / delta_t_cfd))
    print(f'Base env action interval: {base_env.action_interval} (control_dt={delta_t_control}, cfd_dt={delta_t_cfd})')

    print('\n--- Applying Environment Wrappers ---')
    # Wrap 1: Rescale action [-1, 1] to environment's native limits
    # Ensure min/max are numpy arrays of the correct shape (1,)
    min_act = np.array([base_env.action_space.low[0]])
    max_act = np.array([base_env.action_space.high[0]])
    env = CustomRescaleAction(base_env, min_action=min_act, max_action=max_act)
    print(f"Applied CustomRescaleAction. New action space: {env.action_space}")

    # Wrap 2: Delay observations (history buffer)
    # initial_history is now handled outside by running the env
    env = DelayObservationWrapper(env, n_steps=lookback)
    print(f"Applied DelayObservationWrapper (lookback={lookback}). New obs space: {env.observation_space}")
    print('-------------------------------------')

    return env


# Function to safely get data from the info dictionary
def safe_get_from_info(info, key, default=np.nan):
    # Check if info is a dict and key exists
    if isinstance(info, dict) and key in info:
        val = info[key]
        # Handle potential single-element lists/arrays if env returns them
        if isinstance(val, (list, np.ndarray)) and len(val) == 1:
            return val[0]
        elif isinstance(val, (int, float, np.number)):  # np.number covers numpy floats and ints
            return val
        else:
            print(f"Warning: Unexpected type or size for info['{key}']: {val}. Returning default.")
            return default
    return default


def plot_mpc_predictions_vs_actual(time_steps, predicted_force_sequences, actual_forces, results_dir,
                                   plot_suffix):  # Renamed model_id to plot_suffix
    """Plots predicted forces (step 1) vs actual forces."""
    if not predicted_force_sequences or not actual_forces:
        print("Skipping prediction vs actual plot: Missing data.")
        return None

    pred_forces_step1 = np.stack(predicted_force_sequences, axis=0)[:, 0, :]  # Shape (steps, n_forces)
    actual_forces = np.array(actual_forces)  # Shape (steps, 2)

    # Check lengths
    min_len = min(len(pred_forces_step1), len(actual_forces))
    if min_len == 0:
        print("Skipping prediction vs actual plot: Zero length data.")
        return None

    pred_forces_step1 = pred_forces_step1[:min_len]
    actual_forces = actual_forces[:min_len]
    time_steps = time_steps[:min_len]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'MPC: Predicted (t+1) vs Actual Forces ({plot_suffix})')  # Use plot_suffix

    # Plot Cd
    axs[0].plot(time_steps, actual_forces[:, 0], 'k-', linewidth=1.5, label='Actual Cd')
    axs[0].plot(time_steps, pred_forces_step1[:, 0], 'r--', linewidth=1.0, label='Predicted Cd (t+1)')
    axs[0].set_ylabel('Cd')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')

    # Plot Cl
    axs[1].plot(time_steps, actual_forces[:, 1], 'k-', linewidth=1.5, label='Actual Cl')
    axs[1].plot(time_steps, pred_forces_step1[:, 1], 'b--', linewidth=1.0, label='Predicted Cl (t+1)')
    axs[1].set_ylabel('Cl')
    axs[1].set_xlabel('Control Steps')
    axs[1].legend()
    axs[1].grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = results_dir / f'{plot_suffix}_MPC_pred_vs_actual.png'  # Use plot_suffix
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Prediction vs actual plot saved to {plot_filename}")
    return plot_filename


def plot_mpc_optimization_metrics(time_steps, costs, times, results_dir,
                                  plot_suffix):  # Renamed model_id to plot_suffix
    """Plots optimization cost and time per step."""
    if not costs or not times:
        print("Skipping optimization metrics plot: Missing data.")
        return None

    min_len = min(len(costs), len(times))
    if min_len == 0:
        print("Skipping optimization metrics plot: Zero length data.")
        return None

    costs = np.array(costs)[:min_len]
    times = np.array(times)[:min_len]
    time_steps = time_steps[:min_len]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'MPC Optimization Metrics ({plot_suffix})')  # Use plot_suffix

    # Plot Cost
    axs[0].plot(time_steps, costs, 'g-', label='Optimized Cost Value')
    axs[0].set_ylabel('Cost Function Value')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')

    # Plot Time
    axs[1].plot(time_steps, times, 'm-', label='Optimization Time per Step')
    axs[1].set_ylabel('Time (s)')
    axs[1].set_xlabel('Control Steps')
    axs[1].legend()
    axs[1].grid(True, linestyle=':')
    axs[1].set_yscale('log')  # Opt time can vary a lot

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = results_dir / f'{plot_suffix}_MPC_opt_metrics.png'  # Use plot_suffix
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Optimization metrics plot saved to {plot_filename}")
    return plot_filename


def test_controller(controller: MPCControllerLatent, env: gym.Env, results_dir: Path,
                    plot_suffix_for_run: str,
                    sensor_noise_sigma: float = 0.0,
                    noise_seed: int = 42):
    """
    Runs the MPC control loop.

    Parameters
    ----------
    sensor_noise_sigma : float
        Std of additive Gaussian noise in normalised sensor space (sigma = alpha,
        consistent with open-loop noise robustness experiments). 0.0 = clean run.
    noise_seed : int
        Random seed for reproducible noise realisations.
    """

    terminated = False
    # --- Noise RNG and rolling buffer (only active if sensor_noise_sigma > 0) ---
    noise_rng = np.random.default_rng(noise_seed)
    # Noise std in unscaled space: sigma_noise_unscaled = sensor_noise_sigma * p_std
    # This is equivalent to sensor_noise_sigma in normalized space.
    # controller.p_std is a numpy array filled with the scalar p_std value
    noise_sigma_unscaled = sensor_noise_sigma * float(controller.p_std.flat[0])
    # Rolling noise buffer: one noise vector per timestep.
    # Each physical timestep t receives one fixed corruption that persists in all
    # MPC history windows containing t, matching real sensor behavior.
    # Shape is (lookback, n_sensors); n_sensors resolved after env init below.
    noise_buffer = None
    if sensor_noise_sigma > 0.0:
        print(f"Sensor noise ENABLED: sigma={sensor_noise_sigma:.4f} (normalised) "
              f"= {noise_sigma_unscaled:.6f} (unscaled)")
    else:
        print("Sensor noise DISABLED (clean run).")
    actions_applied_unscaled = []  # Store the UNscaled action applied at each step
    rewards = []
    observations_history_unscaled = []  # Store the sequence observation fed to MPC (unscaled sensors)
    predicted_force_sequences = []  # Store the full force prediction horizon at each step
    predicted_control_sequences = []
    actual_forces_list = []  # Store actual forces reported by the env (Cd, Cl)
    optimization_times = []  # Store time taken for controller.optimize
    optimal_costs = []  # Store the cost value associated with the chosen action sequence
    # --------------------------

    # --- Initialize History by Running Env ---
    print(f"\nInitializing environment history for {controller.lookback} steps with zero action...")
    obs_list_init = []
    # Reset env once; DelayObservationWrapper handles internal history init
    # The first obs returned is the lookback sequence filled with the reset state
    obs_seq_init, _ = env.reset()
    # We need the *single* observation from the reset to start the loop
    single_obs_init = obs_seq_init[-1]
    obs_list_init.append(single_obs_init.copy())  # Store the first single observation

    zero_action_rescaled = np.zeros(env.action_space.shape)  # Action in [-1, 1] space

    for i in tqdm(range(controller.lookback - 1), desc="Init Steps", unit="step"):
        # Step with zero action (rescaled to [-1, 1])
        obs_seq, _, term, trunc, _ = env.step(zero_action_rescaled)
        # Store the newest observation from the sequence
        single_step_obs = obs_seq[-1]
        obs_list_init.append(single_step_obs.copy())
        if term or trunc:
            print(
                f"\nWarning: Environment terminated/truncated during initialization step {i + 1}. Resetting and continuing.")
            obs_seq_init, _ = env.reset()
            single_obs_init = obs_seq_init[-1]
            obs_list_init = [single_obs_init.copy()]  # Restart history list

    # Construct the initial state sequences required by the controller
    # obs_seq_unscaled should be the history buffer content AFTER the init steps
    # The wrapper `env` now holds the correct sequence after the loop
    obs_seq_unscaled = env.get_history() # Use helper if available, otherwise need internal access
    if obs_seq_unscaled.shape[0] != controller.lookback:  # Sanity check
        print(
            f"ERROR: Initial observation sequence shape incorrect: {obs_seq_unscaled.shape}. Expected lookback {controller.lookback}")
        raise RuntimeError("History initialization failed.")
    # Initial control history is all zeros (unscaled)
    past_control_unscaled = np.zeros((controller.lookback, controller.action_dim))
    print("Initialization complete.")
    # --- End Initialize History ---

    print("\n--- Starting MPC Control Loop ---")
    # Initialize the rolling noise buffer now that obs_seq_unscaled shape is known
    if sensor_noise_sigma > 0.0:
        n_sensors_for_noise = obs_seq_unscaled.shape[-1]
        noise_buffer = noise_rng.normal(
            0.0, noise_sigma_unscaled,
            size=(controller.lookback, n_sensors_for_noise)
        ).astype(obs_seq_unscaled.dtype)
    else:
        noise_buffer = None
    step_count = 0
    # Add a max step limit to prevent infinite loops
    max_steps = 100000 # adjust as needed
    while not terminated and step_count < max_steps:
        print(f"\n--- MPC Step {step_count} ---")

        # Optimize to get the sequence of future UNscaled controls
        t_start = time.perf_counter()
        try:
            # --- Inject sensor noise using the rolling noise buffer ---
            # Roll the buffer: drop the oldest timestep, sample a new noise vector
            # for the current timestep. This ensures each physical timestep t has
            # one fixed noise realization that persists across all MPC windows that
            # contain t, matching real sensor corruption behavior.
            if sensor_noise_sigma > 0.0 and noise_buffer is not None:
                # Roll out the oldest noise, sample a fresh one for the new timestep
                noise_buffer = np.roll(noise_buffer, -1, axis=0)
                noise_buffer[-1] = noise_rng.normal(
                    0.0, noise_sigma_unscaled,
                    size=(obs_seq_unscaled.shape[-1],)
                ).astype(obs_seq_unscaled.dtype)
                obs_seq_noisy = obs_seq_unscaled + noise_buffer
            else:
                obs_seq_noisy = obs_seq_unscaled
            # Save the observation actually fed to the controller (noisy if applicable)
            observations_history_unscaled.append(obs_seq_noisy.copy())
            # Optimize to get the sequence of future UNscaled controls
            future_controls_unscaled = controller.optimize(obs_seq_noisy, past_control_unscaled)
            predicted_control_sequences.append(future_controls_unscaled.copy())
        except Exception as e:
            print(f"*** ERROR during MPC optimization at step {step_count}: {e}")
            optimization_times.append(np.nan)
            optimal_costs.append(np.nan)
            predicted_control_sequences.append(np.full((controller.horizon, controller.action_dim), np.nan))
            terminated = True
            continue
        t_end = time.perf_counter()
        opt_time = t_end - t_start
        optimization_times.append(opt_time)
        print(f"Optimization time: {opt_time:.3f} s")
        # --------------------

        # --- Log Optimal Cost & Predicted Forces ---
        # Recalculate cost and prediction for the *chosen* sequence
        try:
            with torch.no_grad():
                # For logging cost/prediction, we need to handle sensor slicing correctly if slim encoder is used
                past_sensors_tensor_full_log = torch.tensor(obs_seq_unscaled, dtype=torch.float32,
                                                            device=controller.device)
                if controller.use_slim_encoder and controller.selected_indices is not None:
                    past_sensors_tensor_for_encoder_log = past_sensors_tensor_full_log[:,
                                                          controller.selected_indices]
                else:
                    past_sensors_tensor_for_encoder_log = past_sensors_tensor_full_log

                past_sensors_scaled_tensor_log = (
                                                         past_sensors_tensor_for_encoder_log - controller.p_mean_tensor) / controller.p_std_tensor
                past_sensors_scaled_tensor_log = past_sensors_scaled_tensor_log.unsqueeze(0)

                future_controls_tensor = torch.tensor(future_controls_unscaled, dtype=torch.float32,
                                                      device=controller.device)
                last_applied_action_tensor = torch.tensor(controller.last_applied_action_unscaled, dtype=torch.float32,
                                                          device=controller.device)

                predicted_forces_scaled = controller.predict_horizon_torch(past_sensors_scaled_tensor_log,
                                                                           future_controls_tensor)  # Use prepared scaled tensor
                cost_val_tensor = controller.cost_function_torch(predicted_forces_scaled, future_controls_tensor,
                                                                 last_applied_action_tensor)
                optimal_costs.append(cost_val_tensor.item())
                predicted_forces_unscaled_np = (
                        predicted_forces_scaled * controller.forces_std_tensor + controller.forces_mean_tensor).cpu().numpy()
                predicted_force_sequences.append(predicted_forces_unscaled_np)
        except Exception as e:
            print(f"*** ERROR during final cost/prediction logging at step {step_count}: {e}")
            num_forces = controller.force_decoder.model[-1].out_features
            optimal_costs.append(np.nan)
            predicted_force_sequences.append(np.full((controller.horizon, num_forces), np.nan))

        action_to_apply_unscaled = controller.get_action_to_apply(future_controls_unscaled)

        # --- Commit the first free action to the delay buffer ---
        # With delay d, the action at index d of the optimized sequence is the
        # first truly free action decided *now*; it will be applied d steps
        # from now.  Push it into the FIFO so future MPC calls see it as
        # committed.
        if controller.actuator_delay > 0:
            first_free_action = future_controls_unscaled[controller.actuator_delay]
            controller.commit_action_to_delay_buffer(first_free_action)

        action_rescaled = (action_to_apply_unscaled - controller.limits[0]) / (
                controller.limits[1] - controller.limits[0]) * 2.0 - 1.0
        action_rescaled = np.clip(action_rescaled, -1.0, 1.0)
        try:
            next_obs_seq_unscaled, reward, term, trunc, info = env.step(action_rescaled)
            terminated = term or trunc
        except Exception as e:
            print(f"*** ERROR during environment step {step_count}: {e}")
            terminated = True
            actual_forces_list.append([np.nan, np.nan])
            continue
        if terminated: print(f"Environment terminated or truncated at step {step_count}.")

        actual_cd = safe_get_from_info(info, 'Cd')
        actual_cl = safe_get_from_info(info, 'Cl')
        actual_forces_list.append([actual_cd, actual_cl])
        actions_applied_unscaled.append(action_to_apply_unscaled[0])  # Assuming action_dim is 1
        rewards.append(reward)
        print(
            f"Step {step_count}: Action Applied (Unscaled): {action_to_apply_unscaled[0]:.4f}, Reward: {reward:.4f}, Actual Cd: {actual_cd:.4f}, Actual Cl: {actual_cl:.4f}")

        # Update history for the next iteration
        obs_seq_unscaled = next_obs_seq_unscaled
        past_control_unscaled = np.roll(past_control_unscaled, -1, axis=0)
        past_control_unscaled[-1, :] = action_to_apply_unscaled
        controller.update_last_applied_action(action_to_apply_unscaled)  # Update controller's record of last action

        step_count += 1
        # --- End of Loop ---

    if step_count == max_steps: print(f"Reached maximum step limit ({max_steps}).")
    print("Control loop finished.")

    log_filename = results_dir / f'{plot_suffix_for_run}_MPC_results.h5'  # Use plot_suffix_for_run
    print(f"Saving results to {log_filename}")
    try:
        with h5py.File(log_filename, 'w') as f:
            if observations_history_unscaled: f.create_dataset('observations_history_unscaled',
                                                               data=np.stack(observations_history_unscaled, axis=0))
            if actions_applied_unscaled: f.create_dataset('actions_applied_unscaled',
                                                          data=np.array(actions_applied_unscaled))
            if rewards: f.create_dataset('rewards', data=np.array(rewards))
            if predicted_force_sequences: f.create_dataset('predicted_force_sequences',
                                                           data=np.stack(predicted_force_sequences, axis=0))
            if predicted_control_sequences: f.create_dataset('predicted_control_sequences',
                                                             data=np.stack(predicted_control_sequences, axis=0))
            if actual_forces_list: f.create_dataset('actual_forces', data=np.array(actual_forces_list))
            if optimization_times: f.create_dataset('optimization_times', data=np.array(optimization_times))
            if optimal_costs: f.create_dataset('optimal_costs', data=np.array(optimal_costs))
            f.attrs['sensor_noise_sigma'] = sensor_noise_sigma
            f.attrs['noise_seed'] = noise_seed
            f.attrs['actuator_delay'] = controller.actuator_delay
    except Exception as e:
        print(f"ERROR saving detailed results to HDF5: {e}")
    return actions_applied_unscaled, rewards, predicted_force_sequences, actual_forces_list, optimization_times, optimal_costs


if __name__ == "__main__":
    cwd = Path.cwd()

    # --- Configuration ---
    # ID of the run that produced the core dynamics and force decoder models.
    # Also, the original (full) encoder if USE_SLIM_ENCODER is False.
    original_model_id = '20250906_19_38_58' # This ID is for the base components

    USE_SLIM_ENCODER = True

    # --- Slim Encoder Specific Paths (ONLY USED IF USE_SLIM_ENCODER is True) ---
    # The "case name" under 03_Checkpoints and 04_Results where the slim_folder_name for the desired slim artifacts are located.
    # This MUST match the parent directory of slim_folder_name shown in your actual file path.
    case_name_for_sensor_optim_artifacts = "jet_2Dtruck_20250307_FMsignal_50000" # <<< CORRECT THIS TO MATCH YOUR DATASET NAME

    # The specific directory name of the sensor optimization run (contains phase1/phase2 artifacts).
    # This is the full name of the folder under slim_folder_name.
    slim_run_folder_name = '20250906_19_38_58_LSTM_dim8_lb32_l1_h256_dr0.0_lr0.001_bs512_wC_ntest0_SHAP_Optim_20250906_22_12_02' # <<< CORRECT THIS TO MATCH YOUR RUN
    slim_folder_name = 'shap_optim_runs'

    checkpoints_base = cwd / "03_Checkpoints"
    results_base = cwd / "04_Results"

    if USE_SLIM_ENCODER:
        if not case_name_for_sensor_optim_artifacts or not slim_run_folder_name:
            raise ValueError("If USE_SLIM_ENCODER is True, case_name_for_sensor_optim_artifacts and slim_run_folder_name must be set.")

        # Path to the directory where the slim encoder and selector artifacts for this specific slim_run_folder_name are stored
        slim_artifact_base_ckp_dir = checkpoints_base / case_name_for_sensor_optim_artifacts / slim_folder_name / slim_run_folder_name
        slim_artifact_base_res_dir = results_base / case_name_for_sensor_optim_artifacts / slim_folder_name / slim_run_folder_name

        # Specific filenames (these are from the rank_sensors_shap_and_train_slim.py script's output structure)
        slim_encoder_filename = f"{slim_run_folder_name}_phase2_slim_shap_encoder_slim_shap.pth.tar"
        selected_indices_filename = f"{slim_run_folder_name}_shap_selected_indices.txt"

        SLIM_ENCODER_CKPT_PATH = str(slim_artifact_base_ckp_dir / slim_encoder_filename)
        SELECTED_INDICES_PATH = str(slim_artifact_base_res_dir / selected_indices_filename)

        print(f"Attempting to use Slim Encoder Ckp: {SLIM_ENCODER_CKPT_PATH}")
        print(f"Attempting to use Selected Indices: {SELECTED_INDICES_PATH}")
    # --- End Slim Encoder Config ---

    mpc_horizon = 25
    # ... (rest of the MPC parameters: lr_mpc, num_optim_steps, etc.) ...
    lr_mpc = 0.001
    num_optim_steps = 5
    cost_lambda_rate = 0.0
    cost_lambda_effort = 0.0
    cost_lambda_smoothness = 8.0
    control_limits = [-0.075, 0.075]

    # --- Sensor noise configuration ---
    # Gaussian noise added to sensor observations before passing to the MPC controller.
    # sigma_noise_unscaled = SENSOR_NOISE_SIGMA * p_std, equivalent to SENSOR_NOISE_SIGMA
    # in normalized sensor space, consistent with the open-loop noise robustness experiments.
    # Set to 0.0 for the clean baseline run.
    SENSOR_NOISE_SIGMA = 0.0   # e.g. 0.05, 0.10, 0.20
    NOISE_SEED = 42

    # --- Actuator delay configuration ---
    # Number of control steps of pure input delay (dead time) to model
    # actuator lag.  The MPC freezes the first ACTUATOR_DELAY actions in the
    # prediction horizon to previously committed values.
    # Set to 0 for the baseline (no delay).  Typical values: 0, 1, 2, 3.
    ACTUATOR_DELAY = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        controller = MPCControllerLatent(
            model_identifier=original_model_id, # Use original ID for base components & args
            checkpoints_base_dir=checkpoints_base,
            results_base_dir=results_base,
            horizon=mpc_horizon, limits=control_limits, device=device,
            lr_mpc=lr_mpc, num_optim_steps=num_optim_steps,
            cost_lambda_rate=cost_lambda_rate, cost_lambda_effort=cost_lambda_effort,
            cost_lambda_smoothness=cost_lambda_smoothness,
            use_slim_encoder=USE_SLIM_ENCODER,
            slim_encoder_ckpt_path=SLIM_ENCODER_CKPT_PATH if USE_SLIM_ENCODER else None,
            selected_sensor_indices_path=SELECTED_INDICES_PATH if USE_SLIM_ENCODER else None,
            actuator_delay=ACTUATOR_DELAY
        )
    except Exception as e:
         print(f"\n*** ERROR initializing controller: {e}"); import traceback; traceback.print_exc(); exit(1)

    # --- Setup Environment ---
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_type_suffix = "SlimMPC" if USE_SLIM_ENCODER else "FullMPC"
    noise_suffix = f"_noise{int(SENSOR_NOISE_SIGMA * 100)}pct" if SENSOR_NOISE_SIGMA > 0.0 else ""
    delay_suffix = f"_delay{ACTUATOR_DELAY}" if ACTUATOR_DELAY > 0 else ""
    # Use the controller's actual model_identifier (which might be the original one) for the run name part
    mpc_run_name_suffix = f"{controller.model_identifier}_{run_type_suffix}{noise_suffix}{delay_suffix}_{current_timestamp}"

    try:
        environment_run_config = make_result_dir(time_stamped=False, suffix=mpc_run_name_suffix)
        results_dir_for_this_run = Path(environment_run_config.get("results_dir", "."))
    except NameError:  # Fallback if gymprecice.utils not available
        results_dir_for_this_run = cwd / f"gymprecice-run/{mpc_run_name_suffix}"
        results_dir_for_this_run.mkdir(parents=True, exist_ok=True)
        environment_run_config = {"precice_config": "precice-config.xml", "results_dir": str(results_dir_for_this_run)}
    except Exception as e:  # Catch other potential errors
        print(f"Error with make_result_dir: {e}. Using default results_dir.")
        results_dir_for_this_run = cwd / f"gymprecice-run/{mpc_run_name_suffix}"
        results_dir_for_this_run.mkdir(parents=True, exist_ok=True)
        environment_run_config = {"precice_config": "precice-config.xml", "results_dir": str(results_dir_for_this_run)}

    print(f"Environment results directory for this run: {results_dir_for_this_run}")
    try:
        (results_dir_for_this_run / "gymprecice-run").mkdir(parents=True,
                                                            exist_ok=True)  # Ensure subdir for gymprecice logs
    except:
        pass

    try:
        env = get_env(environment_run_config, controller.lookback)
    except Exception as e:
        print(f"*** ERROR initializing environment: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


    # --- Run MPC Test ---
    actions, rewards = [], []
    predicted_force_sequences, actual_forces_list, optimization_times, optimal_costs = [], [], [], []
    try:
        actions, rewards, predicted_force_sequences, actual_forces_list, \
            optimization_times, optimal_costs = test_controller(
                controller, env, results_dir_for_this_run, mpc_run_name_suffix,
                sensor_noise_sigma=SENSOR_NOISE_SIGMA,
                noise_seed=NOISE_SEED)
    except Exception as e:
        print(f"\n*** ERROR during MPC control loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Closing environment...")
        _ = env.close()  # Added _ = to suppress output if any

    # --- Plot Results ---
    if actions and rewards:
        print("\nPlotting final results...")
        # Truncate all result lists to the shortest length to handle off-by-one
        # errors caused by environment termination mid-step.
        min_len = min(len(rewards), len(actions), len(actual_forces_list),
                      len(predicted_force_sequences), len(optimization_times), len(optimal_costs))
        rewards              = rewards[:min_len]
        actions              = actions[:min_len]
        actual_forces_list   = actual_forces_list[:min_len]
        predicted_force_sequences = predicted_force_sequences[:min_len]
        optimization_times   = optimization_times[:min_len]
        optimal_costs        = optimal_costs[:min_len]
        time_steps = np.arange(min_len)
        # Plot 1: Summary
        fig_summary, ax_summary = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax_summary[0].plot(time_steps, rewards, 'b-', label='Reward')
        ax_summary[0].set_ylabel('Reward')
        ax_summary[0].set_title(f'MPC Perf Summary ({mpc_run_name_suffix})')
        ax_summary[0].grid(True, linestyle=':')
        ax_summary[0].legend(loc='best')
        ax_summary[1].plot(time_steps, actions, 'r-', label='Action Applied (Unscaled)')
        ax_summary[1].set_xlabel('Control Steps')
        ax_summary[1].set_ylabel('Action Value')
        lim_min, lim_max = controller.limits
        padding = 0.1 * (lim_max - lim_min) if (lim_max - lim_min) != 0 else 0.1
        ax_summary[1].set_ylim(lim_min - padding, lim_max + padding)
        ax_summary[1].axhline(lim_min, color='k', linestyle='--', linewidth=0.8)
        ax_summary[1].axhline(lim_max, color='k', linestyle='--', linewidth=0.8)
        ax_summary[1].grid(True, linestyle=':')
        ax_summary[1].legend(loc='best')
        plt.tight_layout()
        plot_filename_summary = results_dir_for_this_run / f'{mpc_run_name_suffix}_summary_plot.png'  # Use suffix
        plt.savefig(plot_filename_summary)
        print(f"Summary plot saved to {plot_filename_summary}")
        plt.close(fig_summary)

        # Plot 2: Predictions vs Actual
        plot_mpc_predictions_vs_actual(time_steps, predicted_force_sequences, actual_forces_list,
                                       results_dir_for_this_run, mpc_run_name_suffix)  # Use suffix
        # Plot 3: Optimization Metrics
        plot_mpc_optimization_metrics(time_steps, optimal_costs, optimization_times, results_dir_for_this_run,
                                      mpc_run_name_suffix)  # Use suffix
    else:
        print("No actions or rewards recorded, skipping plotting.")
    print("\nMPC script finished.")