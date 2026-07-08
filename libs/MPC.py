import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from collections import deque
import json
import h5py

# Import the new models and helper functions
from libs.models import TemporalEncoder, LatentDynamicsModel, ForceDecoder
from parameters import Args

try:
    from libs.test_encoder_predictor import find_checkpoint_files, load_checkpoint
except ImportError:
    print("Could not import from libs.test_encoder_predictor. Assuming helpers are available elsewhere or globally.")


class MPCControllerLatent:
    def __init__(self, model_identifier, checkpoints_base_dir, results_base_dir,
                 horizon, limits, device,
                 lr_mpc=0.1, num_optim_steps=50,
                 cost_lambda_rate=0.01, cost_lambda_effort=0.001, cost_lambda_smoothness=0,
                 use_slim_encoder: bool = False,
                 slim_encoder_ckpt_path: Optional[str] = None,
                 selected_sensor_indices_path: Optional[str] = None,
                 actuator_delay: int = 0):
        """
        MPC Controller using Latent Dynamics Model and Gradient-Based Optimization.

        Args:
            actuator_delay (int): Number of control steps of pure input delay
                (dead time) to model actuator lag. When d > 0, the first d
                actions in the prediction horizon are frozen to the values
                already committed in previous MPC calls (FIFO buffer), and
                only the remaining H - d actions are optimized.  d = 0
                recovers the original zero-delay behaviour.
        """
        self.model_identifier = model_identifier
        self.checkpoints_base_dir = Path(checkpoints_base_dir)
        self.results_base_dir = Path(results_base_dir)
        self.horizon = horizon
        self.limits = np.array(limits)
        self.device = device
        # Gradient-based optimization parameters
        self.lr_mpc = lr_mpc
        self.num_optim_steps = num_optim_steps
        # Cost function weights
        self.cost_lambda_rate = cost_lambda_rate
        self.cost_lambda_effort = cost_lambda_effort
        self.cost_lambda_smoothness = cost_lambda_smoothness

        # --- Actuator delay (dead-time compensation) ---
        self.actuator_delay = actuator_delay
        if self.actuator_delay < 0:
            raise ValueError(f"actuator_delay must be >= 0, got {self.actuator_delay}")
        if self.actuator_delay >= horizon:
            raise ValueError(f"actuator_delay ({self.actuator_delay}) must be < horizon ({horizon})")
        # FIFO buffer of committed actions (will be initialized after action_dim is known)
        self._committed_actions_buffer = None  # deque, length = actuator_delay

        # Slim encoder related attributes
        self.use_slim_encoder = use_slim_encoder
        self.slim_encoder_ckpt_path = Path(slim_encoder_ckpt_path) if slim_encoder_ckpt_path else None
        self.selected_sensor_indices_path = Path(selected_sensor_indices_path) if selected_sensor_indices_path else None
        self.selected_indices = None  # Will be loaded if use_slim_encoder

        # Initialize attributes
        self.encoder = None
        self.dynamics_model = None
        self.force_decoder = None
        self.args = None
        # Store scaling params also as tensors on the correct device
        self.p_mean_tensor, self.p_std_tensor = None, None
        self.control_mean_tensor, self.control_std_tensor = None, None
        self.forces_mean_tensor, self.forces_std_tensor = None, None
        # Keep numpy versions for potential use outside torch graph. Remove if unused.
        self.p_mean, self.p_std = None, None
        self.control_mean, self.control_std = None, None
        self.forces_mean, self.forces_std = None, None

        self.lookback = None
        self.action_dim = None
        self.n_sensors_original = None  # Number of sensors in the original dataset
        self.input_dim_encoder = None  # Actual input dim for the loaded encoder

        self._load_models_and_config()  # Load models and config

        self.prev_cdoptim = None  # store last predicted Cd sequence
        self.prev_cloptim = None  # store last predicted Cl sequence
        self.prev_optim_sequence_unscaled = None  # Store previous sequence
        # Store last applied action (as numpy for interfacing with test_controller)
        self.last_applied_action_unscaled = np.zeros(self.action_dim)
        # Store Cd prediction from the last optimization (for plotting/debug)
        self.Cdoptim = None
        self.Cloptim = None

        # --- Initialize actuator delay FIFO buffer ---
        # Each entry is an action (shape: action_dim) that has been committed
        # but not yet applied to the plant due to actuator dead time.
        # Initialized with zeros (no actuation during the first d steps).
        if self.actuator_delay > 0:
            self._committed_actions_buffer = deque(
                [np.zeros(self.action_dim) for _ in range(self.actuator_delay)],
                maxlen=self.actuator_delay
            )
            print(f"Actuator delay enabled: d = {self.actuator_delay} steps. "
                  f"First {self.actuator_delay} horizon actions will be frozen during optimization.")
        else:
            self._committed_actions_buffer = None
            print("Actuator delay disabled (d = 0).")

    def _find_case_dirs(self):
        """Finds the specific case directories for checkpoints and results."""
        matching_ckp_dirs = []
        matching_res_dirs = []
        for base_dir, matching_list in [(self.checkpoints_base_dir, matching_ckp_dirs),
                                        (self.results_base_dir, matching_res_dirs)]:
            if not base_dir.is_dir(): continue
            for case_dir in base_dir.iterdir():
                if case_dir.is_dir():
                    if "paper_figures" not in case_dir.name and any(self.model_identifier in item.name for item in case_dir.iterdir() if item.is_file()):
                        matching_list.append(case_dir)
        if not matching_ckp_dirs: raise FileNotFoundError(
            f"No checkpoint dir found containing artifacts for {self.model_identifier}")
        if not matching_res_dirs: raise FileNotFoundError(
            f"No results dir found containing artifacts for {self.model_identifier}")

        ckp_dir_to_use = matching_ckp_dirs[0]
        res_dir_to_use = matching_res_dirs[0]
        if len(matching_ckp_dirs) > 1: print(
            f"Warning: Multiple checkpoint dirs found for original model, using {ckp_dir_to_use}")
        if len(matching_res_dirs) > 1: print(
            f"Warning: Multiple results dirs found for original model, using {res_dir_to_use}")
        return ckp_dir_to_use, res_dir_to_use

    def _load_models_and_config(self):
        """Loads models, args from the latent space file, and scaling parameters from the reference training datafile."""
        print("--- Loading MPC Models and Configuration ---")
        case_ckp_dir_orig, case_results_dir_orig = self._find_case_dirs()

        latent_file_pattern = f"{self.model_identifier}*latent_space.hdf5"
        latent_files = sorted(list(case_results_dir_orig.glob(latent_file_pattern)))
        latent_file = next((f for f in latent_files if '_eval' not in f.name),
                           latent_files[0] if latent_files else None)
        if not latent_file: raise FileNotFoundError(f"Latent file not found in {case_results_dir_orig}")

        print(f"Loading args from: {latent_file}")
        try:
            with h5py.File(latent_file, 'r') as f:
                args_json = f.attrs['args']
                args_dict = json.loads(args_json)
                self.args = Args(**args_dict)
        except Exception as e:
            print(f"Error loading args from latent space file: {e}")
            raise

        # --- Load scaling parameters from the latent space file ---
        # This ensures the MPC uses the *exact* scaling the model was trained with.
        try:
            print(f"Loading scaling parameters from: {latent_file}")
            with h5py.File(latent_file, 'r') as f:
                p_mean_np_full = f['p_mean'][:]
                p_std_scalar = f['p_std'][()] # Assuming it's a scalar
                forces_mean_np = f['forces_mean'][:]
                forces_std_np = f['forces_std'][:]
                control_mean_np = f['control_mean'][:]
                control_std_np = f['control_std'][:]

            if not (np.isscalar(p_std_scalar) or p_std_scalar.ndim == 0):
                raise ValueError(
                    f"Pstd loaded from latent file ('{latent_file}') must be a scalar, but has shape: {p_std_scalar.shape}")
            self.n_sensors_original = p_mean_np_full.shape[0]
        except Exception as e:
            print(f"Error loading scaling parameters from latent space file: {e}")
            raise

        self.lookback = self.args.lookback
        self.action_dim = control_mean_np.shape[0] if hasattr(control_mean_np, 'shape') else 1

        self.control_mean_tensor = torch.tensor(control_mean_np, dtype=torch.float32, device=self.device)
        self.control_std_tensor = torch.tensor(control_std_np, dtype=torch.float32, device=self.device)
        self.forces_mean_tensor = torch.tensor(forces_mean_np, dtype=torch.float32, device=self.device)
        self.forces_std_tensor = torch.tensor(forces_std_np, dtype=torch.float32, device=self.device)
        self.control_mean, self.control_std = control_mean_np, control_std_np
        self.forces_mean, self.forces_std = forces_mean_np, forces_std_np

        model_id_for_dynamics_force_ckpts = self.args.modelname if hasattr(self.args,
                                                                           'modelname') and self.args.modelname else self.model_identifier

        if self.use_slim_encoder:
            print("--- Configuring for SLIM ENCODER ---")
            if not self.slim_encoder_ckpt_path or not self.slim_encoder_ckpt_path.is_file():
                raise FileNotFoundError(f"Slim encoder checkpoint not found: {self.slim_encoder_ckpt_path}")
            if not self.selected_sensor_indices_path or not self.selected_sensor_indices_path.is_file():
                raise FileNotFoundError(f"Selected sensor indices file not found: {self.selected_sensor_indices_path}")

            self.selected_indices = np.loadtxt(self.selected_sensor_indices_path, dtype=int)
            print(f"Loaded {len(self.selected_indices)} selected sensor indices.")

            self.input_dim_encoder = len(self.selected_indices)
            p_mean_np_selected = p_mean_np_full[self.selected_indices]
            p_std_np_selected_array = np.full(len(self.selected_indices), p_std_scalar)

            self.p_mean_tensor = torch.tensor(p_mean_np_selected, dtype=torch.float32, device=self.device)
            self.p_std_tensor = torch.tensor(p_std_scalar, dtype=torch.float32, device=self.device)
            self.p_mean, self.p_std = p_mean_np_selected, p_std_np_selected_array

            print(f"Slim Encoder: InputDim={self.input_dim_encoder}, Lookback={self.lookback}")
            encoder_dropout = self.args.dropout if hasattr(self.args, 'dropout') else 0.0
            self.encoder = TemporalEncoder(input_dim=self.input_dim_encoder, latent_dim=self.args.latent_dim,
                                           hidden_dim=self.args.enc_hidden_dim, num_layers=self.args.n_layers,
                                           dropout_rate=encoder_dropout).to(self.device)
            load_checkpoint(self.encoder, self.slim_encoder_ckpt_path, self.device)
            print(f"Loaded SLIM encoder weights from: {self.slim_encoder_ckpt_path}")
        else:
            print("--- Configuring for ORIGINAL ENCODER ---")
            self.input_dim_encoder = self.n_sensors_original
            self.selected_indices = None
            p_std_np_full_array = np.full(self.n_sensors_original, p_std_scalar)

            self.p_mean_tensor = torch.tensor(p_mean_np_full, dtype=torch.float32, device=self.device)
            self.p_std_tensor = torch.tensor(p_std_scalar, dtype=torch.float32, device=self.device)
            self.p_mean, self.p_std = p_mean_np_full, p_std_np_full_array

            encoder_ckpt_orig, _, _ = find_checkpoint_files(case_ckp_dir_orig, model_id_for_dynamics_force_ckpts)
            if not encoder_ckpt_orig: raise FileNotFoundError("Original encoder checkpoint not found.")

            print(f"Original Encoder: InputDim={self.input_dim_encoder}, Lookback={self.lookback}")
            encoder_dropout = self.args.dropout if hasattr(self.args, 'dropout') else 0.0
            self.encoder = TemporalEncoder(input_dim=self.input_dim_encoder, latent_dim=self.args.latent_dim,
                                           hidden_dim=self.args.enc_hidden_dim, num_layers=self.args.n_layers,
                                           dropout_rate=encoder_dropout).to(self.device)
            load_checkpoint(self.encoder, encoder_ckpt_orig, self.device)
            print(f"Loaded ORIGINAL encoder weights from: {encoder_ckpt_orig}")

        _, dynamics_ckpt, forces_ckpt = find_checkpoint_files(case_ckp_dir_orig, model_id_for_dynamics_force_ckpts)
        if not dynamics_ckpt or not forces_ckpt:
            raise FileNotFoundError("Dynamics or Force decoder checkpoint for original model not found.")

        print("Initializing models (Dynamics & Force Decoder)...")
        use_residual = self.args.residual_predictor if hasattr(self.args, 'residual_predictor') else False
        self.dynamics_model = LatentDynamicsModel(latent_dim=self.args.latent_dim, action_dim=self.action_dim,
                                                  hidden_dim=self.args.dyn_hidden_dim, use_residual=use_residual).to(
            self.device)
        forces_hidden = self.args.forces_hidden_dim if hasattr(self.args, 'forces_hidden_dim') else 64
        forces_dropout = self.args.forces_dropout if hasattr(self.args, 'forces_dropout') else 0.0
        decoder_arch = self.args.force_decoder_arch if hasattr(self.args, 'force_decoder_arch') else "FCN"
        self.force_decoder = ForceDecoder(latent_dim=self.args.latent_dim, hidden_dim=forces_hidden,
                                          dropout_rate=forces_dropout, arch=decoder_arch).to(self.device)

        print("Loading model weights (Dynamics & Force Decoder)...")
        load_checkpoint(self.dynamics_model, dynamics_ckpt, self.device)
        load_checkpoint(self.force_decoder, forces_ckpt, self.device)

        self.encoder.eval()
        self.dynamics_model.eval()
        self.force_decoder.eval()
        print("--- MPC Models Loaded and Ready ---")

    def optimize(self, past_sensors_unscaled, past_control_unscaled):
        """
        Optimizes the control action sequence using PyTorch Adam optimizer.

        When actuator_delay d > 0, the first d actions in the horizon are
        frozen to the values stored in the committed-actions FIFO buffer
        (actions that have been decided in previous MPC calls but have not
        yet taken effect due to actuator dead time).  Only the remaining
        H - d actions are optimized.
        """
        d = self.actuator_delay  # shorthand

        past_sensors_tensor_full = torch.tensor(past_sensors_unscaled, dtype=torch.float32,
                                                device=self.device)

        if self.use_slim_encoder and self.selected_indices is not None:
            past_sensors_tensor_for_encoder = past_sensors_tensor_full[:, self.selected_indices]
        else:
            past_sensors_tensor_for_encoder = past_sensors_tensor_full

        past_sensors_scaled_tensor = (past_sensors_tensor_for_encoder - self.p_mean_tensor) / self.p_std_tensor
        past_sensors_scaled_tensor = past_sensors_scaled_tensor.unsqueeze(0)

        last_applied_action_tensor = torch.tensor(self.last_applied_action_unscaled, dtype=torch.float32,
                                                  device=self.device)

        # --- Build the frozen prefix from the committed-actions buffer ---
        if d > 0:
            committed_np = np.array(list(self._committed_actions_buffer))  # (d, action_dim)
            committed_tensor = torch.tensor(committed_np, dtype=torch.float32,
                                            device=self.device)  # frozen, no grad
        else:
            committed_tensor = None

        # --- Initial guess for the FREE (optimizable) portion: H - d actions ---
        n_free = self.horizon - d

        if self.prev_optim_sequence_unscaled is not None and self.prev_optim_sequence_unscaled.shape[0] == self.horizon:
            # Shift the previous full sequence left by 1 (warm start)
            prev_shifted = np.roll(self.prev_optim_sequence_unscaled, -1, axis=0)
            prev_shifted[-1] = prev_shifted[-2]
            # Take only the free portion (indices d: onward)
            initial_guess_free = prev_shifted[d:]
        else:
            initial_guess_free = self.last_applied_action_unscaled.repeat(n_free).reshape(n_free,
                                                                                          self.action_dim)

        # Only the free actions require gradients
        u_free_tensor = torch.tensor(initial_guess_free, dtype=torch.float32,
                                     device=self.device, requires_grad=True)
        with torch.no_grad():
            u_free_tensor.clamp_(self.limits[0], self.limits[1])

        optimizer = torch.optim.Adam([u_free_tensor], lr=self.lr_mpc)

        self.encoder.train()
        self.dynamics_model.train()
        self.force_decoder.train()

        print(
            f"Starting Adam MPC optimization (lr={self.lr_mpc}, steps={self.num_optim_steps}, "
            f"actuator_delay={d}, free_actions={n_free})...")

        for i in range(self.num_optim_steps):
            optimizer.zero_grad()

            # Assemble the full horizon sequence: [committed (frozen) | free (optimizable)]
            if committed_tensor is not None:
                u_seq_full = torch.cat([committed_tensor, u_free_tensor], dim=0)
            else:
                u_seq_full = u_free_tensor

            predicted_forces_scaled = self.predict_horizon_torch(past_sensors_scaled_tensor,
                                                                 u_seq_full)
            cost = self.cost_function_torch(predicted_forces_scaled, u_seq_full,
                                            last_applied_action_tensor)
            cost.backward()
            optimizer.step()
            with torch.no_grad():
                u_free_tensor.clamp_(self.limits[0], self.limits[1])

            print(f"  Optim Step {i + 1}/{self.num_optim_steps}, Cost: {cost.item():.4f}")

        self.encoder.eval()
        self.dynamics_model.eval()
        self.force_decoder.eval()

        print("Optimization finished.")

        # --- Assemble the final full-horizon sequence for logging/plotting ---
        with torch.no_grad():
            if committed_tensor is not None:
                final_u_seq_tensor = torch.cat([committed_tensor, u_free_tensor.detach()], dim=0)
            else:
                final_u_seq_tensor = u_free_tensor.detach()

        # Build the initial_guess for plotting (full horizon)
        if d > 0:
            initial_guess = np.concatenate([committed_np, initial_guess_free], axis=0)
        else:
            initial_guess = initial_guess_free

        with torch.no_grad():
            # Recalculate forces and cost for the chosen best sequence for logging/plotting
            final_forces_scaled = self.predict_horizon_torch(past_sensors_scaled_tensor, final_u_seq_tensor)
            final_cost = self.cost_function_torch(final_forces_scaled, final_u_seq_tensor,
                                                  last_applied_action_tensor)
            print(f"Final Optimized Cost: {final_cost.item():.4f}")
            # Now, convert the final tensor to a numpy array for plotting and returning
            u_seq_optimized_unscaled = final_u_seq_tensor.cpu().numpy()
            # --- Debug Plotting ---
            try:
                # If figure 'plot_mpc_debug' exists, clear it for new data. Otherwise, it will be created.
                fig_debug = plt.figure('plot_mpc_debug')  # Get existing or create new
                fig_debug.clf()  # Clear the figure
            except:  # Fallback if plt.figure with num fails for some reason (should not with string name)
                fig_debug, _ = plt.subplots(3, 1, sharex=True, num='plot_mpc_debug', figsize=(10, 6))

            ax_debug = fig_debug.subplots(3, 1, sharex=True)
            if not isinstance(ax_debug, np.ndarray):
                ax_debug = np.array([ax_debug])

            fig_debug.suptitle('MPC Optimization Result')

            ax_debug[0].plot(np.arange(self.horizon), u_seq_optimized_unscaled, 'b-o', markersize=3,
                             label='Optimized Action (Unscaled)')
            ax_debug[0].plot(np.arange(self.horizon), initial_guess, 'g--', markersize=2, label='Initial guess')
            ax_debug[0].set_ylabel('Action Value')
            lim_pad = 0.1 * (self.limits[1] - self.limits[0]) if (self.limits[1] - self.limits[0]) != 0 else 0.1
            ax_debug[0].set_ylim(self.limits[0] - lim_pad, self.limits[1] + lim_pad)
            ax_debug[0].axhline(self.limits[0], color='k', linestyle='--', linewidth=0.8)
            ax_debug[0].axhline(self.limits[1], color='k', linestyle='--', linewidth=0.8)
            ax_debug[0].grid(True, linestyle=':')
            ax_debug[0].legend(loc='upper right')

            if self.Cdoptim is not None:
                plot_horizon = min(self.horizon, len(self.Cdoptim))
                ax_debug[1].plot(np.arange(plot_horizon), self.Cdoptim[:plot_horizon], 'r-s', markersize=3,
                                 label='Predicted Cd (Final Optim)')
                if self.prev_cdoptim is not None and len(self.prev_cdoptim) > 1:
                    prev_cdoptim_shifted = self.prev_cdoptim[1:]
                    prev_plot_len = min(plot_horizon - 1, len(prev_cdoptim_shifted))
                    if prev_plot_len > 0:
                        ax_debug[1].plot(np.arange(prev_plot_len), prev_cdoptim_shifted[:prev_plot_len], 'k--',
                                         markersize=2, label='Previous predicted Cd')

                ax_debug[1].set_ylabel('Predicted Cd')
                ax_debug[1].grid(True, linestyle=':')
                ax_debug[1].legend(loc='upper right')

            if self.Cloptim is not None:
                plot_horizon = min(self.horizon, len(self.Cloptim))
                ax_debug[2].plot(np.arange(plot_horizon), self.Cloptim[:plot_horizon], 'm-s', markersize=3,
                                 label='Predicted Cl (Final Optim)')
                if self.prev_cloptim is not None and len(self.prev_cloptim) > 1:
                    prev_cloptim_shifted = self.prev_cloptim[1:]
                    prev_plot_len = min(plot_horizon - 1, len(prev_cloptim_shifted))
                    # This block was missing in the previous version
                    if prev_plot_len > 0:
                        ax_debug[2].plot(np.arange(prev_plot_len), prev_cloptim_shifted[:prev_plot_len], 'k--',
                                         markersize=2, label='Previous predicted Cl')

                ax_debug[2].set_ylabel('Predicted Cl')
                ax_debug[2].grid(True, linestyle=':')
                ax_debug[2].legend(loc='upper right')

            ax_debug[2].set_xlabel('Prediction Horizon Steps')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            fig_debug.canvas.draw()  # Explicitly draw the canvas
            fig_debug.canvas.flush_events()  # Process GUI events
            plt.pause(0.1)  # Keep pause to allow time for rendering and viewing
            # --- End Debug Plotting ---

        self.prev_cdoptim = self.Cdoptim
        self.prev_cloptim = self.Cloptim
        self.prev_optim_sequence_unscaled = u_seq_optimized_unscaled.copy()

        return u_seq_optimized_unscaled

    def predict_horizon_torch(self, past_sensors_scaled_tensor, future_control_unscaled_tensor):
        """
        Predicts forces over the horizon using PyTorch models and tensors.
        """
        predicted_forces_scaled_list = []
        z_current = self.encoder(past_sensors_scaled_tensor)
        future_control_scaled_tensor = (
                                               future_control_unscaled_tensor - self.control_mean_tensor) / self.control_std_tensor

        for i in range(self.horizon):
            a_current = future_control_scaled_tensor[i].unsqueeze(0)
            z_next = self.dynamics_model(z_current, a_current)
            forces_scaled_next = self.force_decoder(z_next)
            predicted_forces_scaled_list.append(forces_scaled_next.squeeze(0))
            z_current = z_next

        predicted_forces_scaled_tensor = torch.stack(predicted_forces_scaled_list, dim=0)
        return predicted_forces_scaled_tensor

    def cost_function_torch(self, predicted_forces_scaled_tensor, future_control_unscaled_tensor,
                            last_applied_action_unscaled_tensor):
        """
        Calculates the MPC cost using PyTorch tensors.
        """
        predicted_forces_unscaled = predicted_forces_scaled_tensor * self.forces_std_tensor + self.forces_mean_tensor

        # Store Cd for potential analysis (detach from graph)
        self.Cdoptim = predicted_forces_unscaled[:, 0].detach().cpu().numpy()
        self.Cloptim = predicted_forces_unscaled[:, 1].detach().cpu().numpy()

        # --- Cost Calculation using PyTorch ops ---
        baseline_cd = 1.051  # TODO: Configurable tensor?
        cd_predictions = predicted_forces_unscaled[:, 0]

        # Amplitude Penalty
        cd_amplitude_penalty = torch.tensor(0.0, device=self.device)
        if self.horizon > 1:
            cd_amplitude_penalty = (torch.max(cd_predictions) - torch.min(cd_predictions)) * 0.1

        # Mean Increment Penalty
        cd_mean_increment = torch.mean(cd_predictions) - baseline_cd

        # Cl Penalty
        cl_mean_abs_penalty = torch.tensor(0.0, device=self.device)
        if predicted_forces_unscaled.shape[1] > 1:
            cl_mean_abs_penalty = torch.mean(torch.abs(predicted_forces_unscaled[:, 1])) * 0.01

        # Action Rate Penalty (First step)
        action_rate_penalty = torch.tensor(0.0, device=self.device)
        if self.horizon > 0:
            action_diff = future_control_unscaled_tensor[0] - last_applied_action_unscaled_tensor
            action_rate_penalty = self.cost_lambda_rate * torch.sum(action_diff ** 2)

        # Control Effort Penalty (Mean squared over horizon)
        control_effort_penalty = self.cost_lambda_effort * torch.mean(future_control_unscaled_tensor ** 2)

        # Control Smoothness Penalty
        smoothness_penalty = torch.tensor(0.0, device=self.device)
        if self.horizon > 1:
            # Calculate difference between consecutive actions in the sequence
            # diff(u, axis=0) -> u[1]-u[0], u[2]-u[1], ...
            action_diffs_seq = torch.diff(future_control_unscaled_tensor, dim=0)
            # Penalize the mean squared difference
            smoothness_penalty = self.cost_lambda_smoothness * torch.mean(action_diffs_seq ** 2)

        # Total Cost
        cost = (cd_mean_increment + cd_amplitude_penalty + cl_mean_abs_penalty +
                action_rate_penalty + control_effort_penalty + smoothness_penalty)

        return cost

    def update_last_applied_action(self, applied_action_unscaled):
        """Stores the last action that was actually applied to the environment."""
        self.last_applied_action_unscaled = np.array(applied_action_unscaled).reshape(self.action_dim)

    def commit_action_to_delay_buffer(self, action_unscaled):
        """
        Push a newly decided action into the committed-actions FIFO buffer.

        Called from the control loop *after* optimize(). The action at index
        ``actuator_delay`` of the optimized horizon (i.e. the first truly free
        action) is the one that will be applied ``d`` steps from now, so it is
        committed here.  The oldest entry is automatically popped by the deque.
        """
        if self._committed_actions_buffer is not None:
            self._committed_actions_buffer.append(
                np.array(action_unscaled).reshape(self.action_dim)
            )

    def get_action_to_apply(self, optimized_sequence_unscaled):
        """
        Returns the action that should actually be sent to the plant at this
        MPC step, accounting for actuator delay.

        With delay d = 0 this is simply optimized_sequence_unscaled[0].
        With delay d > 0, the action applied now is the oldest entry in the
        committed-actions buffer (decided d steps ago), which sits at index 0
        of the full optimized horizon (the frozen prefix).
        """
        return optimized_sequence_unscaled[0].copy()