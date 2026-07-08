import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # For detailed metrics
from torch.utils.data import DataLoader, TensorDataset # For potential batch prediction

from libs.models import TemporalEncoder, LatentDynamicsModel, ForceDecoder
from libs.data import get_prepared_data, loadData
from parameters import Args
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})


def load_checkpoint(model, checkpoint_path, device, verbose=True):
    """Load model parameters from checkpoint."""
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
    elif not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


def load_latents_file(path, printer=False):
    with h5py.File(path, 'r') as f:
        latent_space = f['latent_space'][:]
        # The new format saves sequences. We need the force and control from the first step of each sequence.
        if 'c_seq_all' in f:
            forces_scaled = f['c_seq_all'][:, 0, :]
        else:
            raise KeyError("HDF5 file must contain 'c_seq_all' dataset.")

        if 'a_seq_all' in f:
            control_scaled = f['a_seq_all'][:, 0, :]
        else:
            raise KeyError("HDF5 file must contain 'a_seq_all' dataset.")

        # Correctly load scalar or array data
        forces_mean = f['forces_mean'][()] if len(f['forces_mean'].shape) == 0 else f['forces_mean'][:]
        forces_std = f['forces_std'][()] if len(f['forces_std'].shape) == 0 else f['forces_std'][:]
        control_mean = f['control_mean'][()] if len(f['control_mean'].shape) == 0 else f['control_mean'][:]
        control_std = f['control_std'][()] if len(f['control_std'].shape) == 0 else f['control_std'][:]

        # Load the JSON string for args and convert to dictionary
        args_json = f.attrs['args']
        args_dict = json.loads(args_json)
        args_instance = Args(**args_dict)

    if printer:
        print('latent_space: ', latent_space.shape, latent_space.dtype)
        print('forces_scaled: ', forces_scaled.shape, forces_scaled.dtype)
        # More robust printing for potentially scalar values
        print('forces_mean: ', forces_mean.shape if hasattr(forces_mean, 'shape') else 'scalar', forces_mean)
        print('forces_std: ', forces_std.shape if hasattr(forces_std, 'shape') else 'scalar', forces_std)
        print('control_scaled: ', control_scaled.shape, control_scaled.dtype)
        print('control_mean: ', control_mean.shape if hasattr(control_mean, 'shape') else 'scalar', control_mean)
        print('control_std: ', control_std.shape if hasattr(control_std, 'shape') else 'scalar', control_std)

    return latent_space, forces_scaled, forces_mean, forces_std, control_scaled, control_mean, control_std, args_instance


def evaluate_models_over_time(encoder, dynamics_model, forces_decoder, dataloader, device):
    """
    Evaluate the predictor by computing the standardized MSE error between
    predicted and real latent states over time for each sequence.
    The error is standardized by dividing by the variance of the latent space.
    """
    encoder.eval()
    dynamics_model.eval()
    all_errors = []
    all_indices = []  # Keep track of sequence indices

    # First, compute the standard deviation of the latent space
    latent_vectors = []

    # Collect latent vectors to compute standard deviation
    with torch.no_grad():
        for batch_idx, (s_t, _, _, _) in enumerate(dataloader):
            s_t = s_t.to(device)
            z_t = encoder(s_t)
            latent_vectors.append(z_t.cpu())

    # Concatenate all latent vectors and compute standard deviation per dimension
    all_latent_vectors = torch.cat(latent_vectors, dim=0)
    latent_std = torch.std(all_latent_vectors, dim=0)

    # Avoid division by zero
    latent_std = torch.clamp(latent_std, min=1e-8)

    # Now compute standardized errors
    with torch.no_grad():
        for batch_idx, (s_t, a_t, s_t1, c_t) in enumerate(dataloader):
            # Move to device
            s_t = s_t.to(device)
            a_t = a_t.to(device)
            s_t1 = s_t1.to(device)
            c_t = c_t.to(device)

            # Compute current latent state and real next latent state
            z_t = encoder(s_t)  # latent representation at time t
            z_t1_real = encoder(s_t1)  # latent representation at time t+1 (ground truth)
            # Predict next latent state using the dynamics model
            # collapse the extra  lookforward  dimension so action is (batch,1)
            a_current = a_t[:, 0, :]  # shape: (batch, control_dim)
            z_t1_pred = dynamics_model(z_t, a_current)

            # Compute standardized error for each sample in the batch
            # Divide squared differences by variance (std^2) of each dimension
            squared_diff = (z_t1_pred - z_t1_real) ** 2
            standardized_squared_diff = squared_diff / (latent_std.to(device) ** 2)

            # Take mean across dimensions to get standardized MSE
            batch_errors = torch.mean(standardized_squared_diff, dim=1).cpu().numpy()

            all_errors.append(batch_errors)
            all_indices.append(np.arange(len(batch_errors)) + batch_idx * dataloader.batch_size)

    all_errors = np.concatenate(all_errors)
    all_indices = np.concatenate(all_indices)
    return all_errors, all_indices


import math

def plot_pred_sample(true, pred, output_path, series_to_plot=None):
    """
    Plot selected true and predicted time series in a two-column layout.

    Parameters:
        true (np.ndarray): Array of true values with shape (n_series, n_time_steps).
        pred (np.ndarray): Array of predicted values with shape (n_series, n_time_steps).
        output_path (str): File path to save the generated plot.
        series_to_plot (list, optional): List of indices to specify which time series to plot.
                                          Defaults to the first 8 series if available.
    """
    # Default to first n series if not specified
    if series_to_plot is None:
        series_to_plot = list(range(min(8, true.shape[1])))

    num_series = len(series_to_plot)
    if num_series == 0:
        print("No series to plot. Skipping plot generation.")
        return None

    n_cols = 2
    n_rows = math.ceil(num_series / n_cols)

    # Create subplots in a two-column layout
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 1. * n_rows), sharex=True, squeeze=False)
    axs_flat = axs.flatten()

    # Plot each selected time series
    for i, idx in enumerate(series_to_plot):
        ax = axs_flat[i]
        # Plotting the data
        ax.plot(true[:, idx], color='black', linewidth=1., label='Reference')
        ax.plot(pred[:, idx], color='tab:blue', linestyle='--', linewidth=1., label='Predicted')

        # Set a clean title using LaTeX for subscript
        ax.set_title(f"$z_{{{idx+1}}}$", loc='left', fontsize='medium')

        # Clean look: remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Subtle grid for readability
        ax.grid(True, linestyle=':', alpha=0.6)

        # Set x-label only for the plots in the bottom row
        current_row = i // n_cols
        if current_row == n_rows - 1:
            ax.set_xlabel("$t_c$")

    # Add a single, clean legend for the entire figure
    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)

    # Turn off any unused subplots if the number of series is odd
    for i in range(num_series, len(axs_flat)):
        axs_flat[i].axis('off')

    # Adjust layout to prevent title overlap and fit legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure in multiple formats
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(Path(output_path).with_suffix('.pdf'), format='pdf')
    plt.savefig(Path(output_path).with_suffix('.eps'), format='eps')
    plt.close()
    print(f"Saved prediction sample plot to {output_path}")

    return output_path


def plot_error_over_time(errors, indices, output_path):
    """Plot the error over time."""
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Plot errors on the subplot
    window_size = 20
    smoothed_errors = np.convolve(errors, np.ones(window_size) / window_size, mode='valid')
    smoothed_indices = indices[window_size - 1:]

    ax1.plot(indices, errors, linestyle='-', color='tab:blue', label='Raw', linewidth=0.5)
    ax1.plot(smoothed_indices, smoothed_errors, linestyle='-', color='k', linewidth=1.5, label='Smoothed')
    #ax1.set_ylabel('One-Step Latent Prediction Error (Normalized MSE)')
    ax1.set_title('One-step latent prediction error (Normalised MSE)')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.set_xlabel('t')

    # Finalize plot
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(Path(output_path).with_suffix('.pdf'), format='pdf')
    plt.savefig(Path(output_path).with_suffix('.eps'), format='eps')
    plt.close()
    print(f"Saved error over time plot to {output_path}")

    return output_path


def evaluate_forces_decoder(encoder, forces_decoder, dataloader, device, forces_mean, forces_std):
    """
    Evaluate the forces decoder by computing the ABSOLUTE error between
    predicted (from z_t) and real forces (c_t) over time for each sequence.
    Unscales using provided mean/std (should be from training).
    """
    encoder.eval()
    forces_decoder.eval()
    all_errors_list = []
    all_indices = []

    forces_mean_np = np.array(forces_mean)
    forces_std_np = np.array(forces_std)
    forces_mean_tensor = torch.tensor(forces_mean_np, device=device, dtype=torch.float32)
    forces_std_tensor = torch.tensor(forces_std_np, device=device, dtype=torch.float32)

    with torch.no_grad():
        current_idx_base = 0
        for s_t, a_seq, s_t1, c_t_scaled in dataloader:
            s_t = s_t.to(device)
            c_t_target_scaled = c_t_scaled.to(device)
            batch_size_current = s_t.shape[0]

            # --- Get Target Forces ---
            # Handle different possible shapes for c_t_target_scaled based on dataloader prep
            if c_t_target_scaled.ndim == 3:
                # If ndim is 3, assume shape (batch, lookforward, 2)
                # For evaluation, we typically compare against the *first* step's target force
                # if lookforward was 1, or the *last* step's target force if lookforward > 1
                # Let's consistently use the first step's target for this eval function.
                c_t_target_scaled = c_t_target_scaled[:, 0, :] # Select first time step -> shape (batch, 2)
            elif c_t_target_scaled.ndim == 2:
                # Already in (batch, 2) format, assume this corresponds to the first step
                pass
            else:
                 # Handle unexpected shapes
                 print(f"Error: Unexpected shape for c_t_scaled: {c_t_target_scaled.shape}. Skipping batch.")
                 current_idx_base += batch_size_current # Important to advance index even on skip
                 continue # Skip batch if shape is wrong


            # --- Rest of the function remains the same ---

            # 1. Compute current latent state from input sequence s_t
            z_t = encoder(s_t) # Input: (batch, lookback, features) -> Output: (batch, latent_dim)

            # 2. Predict force coefficients from z_t
            c_t_pred_scaled = forces_decoder(z_t) # Shape: (batch, 2)

            # 3. Unscale predictions and targets using TRAINING mean/std
            c_t_pred_unscaled = c_t_pred_scaled * forces_std_tensor + forces_mean_tensor
            c_t_target_unscaled = c_t_target_scaled * forces_std_tensor + forces_mean_tensor

            # 4. Compute absolute error
            batch_errors = torch.abs(c_t_target_unscaled - c_t_pred_unscaled) # Shape: (batch, 2)

            # Append the numpy array for this batch to the list
            all_errors_list.append(batch_errors.cpu().numpy()) # Store (batch_size_current, 2) arrays

            # Store indices
            all_indices.append(np.arange(current_idx_base, current_idx_base + batch_size_current))
            current_idx_base += batch_size_current # Advance index

    if not all_errors_list:
        print("Warning: No force errors were calculated.")
        return np.array([]).reshape(0, 2), np.array([])

    all_errors = np.concatenate(all_errors_list, axis=0)
    all_indices = np.concatenate(all_indices)

    return all_errors, all_indices


def plot_forces_error_over_time(errors, indices, output_path, forces=None):
    """Plot smoothed errors for Cd and Cl in two separate subplots and actual forces in a third subplot."""
    import matplotlib.pyplot as plt

    # If errors is 2D, assume shape (num_samples, 2) with first column for Cd and second for Cl.
    if errors.ndim != 2 or errors.shape[1] != 2:
        raise ValueError("Expected errors to be a 2D array with 2 columns (Cd and Cl errors).")

    # Smooth each error series using a moving average (window size = 20)
    window_size = 20
    smoothed_cd = np.convolve(errors[:, 0], np.ones(window_size) / window_size, mode='valid')
    smoothed_cl = np.convolve(errors[:, 1], np.ones(window_size) / window_size, mode='valid')
    smoothed_indices = indices[window_size - 1:]

    # Create a figure with 3 vertically stacked subplots sharing the x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Top subplot: Smoothed error for Cd
    ax1.plot(indices, errors[:, 0], linestyle='-', color='green', alpha=0.5, label='Raw error Cd', linewidth=0.5)
    ax1.plot(smoothed_indices, smoothed_cd, linestyle='-', color='green', linewidth=2, label='Smoothed error Cd')
    ax1.set_ylabel('Error (Cd)')
    ax1.set_title('Cd Error Over Time')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Middle subplot: Smoothed error for Cl
    ax2.plot(indices, errors[:, 1], linestyle='-', color='orange', alpha=0.5, label='Raw error Cl', linewidth=0.5)
    ax2.plot(smoothed_indices, smoothed_cl, linestyle='-', color='orange', linewidth=2, label='Smoothed Error Cl')
    ax2.set_ylabel('Error (Cl)')
    ax2.set_title('Cl Error Over Time')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Bottom subplot: Actual force coefficients
    if forces is not None:
        ax3.plot(forces[:, 0], linestyle='-', color='green', alpha=0.7, label='Cd', linewidth=0.5)
        ax3.plot(forces[:, 1], linestyle='-', color='orange', alpha=0.7, label='Cl', linewidth=0.5)
    ax3.set_xlabel('Sequence index (Sample)')
    ax3.set_ylabel('Force coefficients')
    ax3.set_title('Actual forces (Cd \& Cl)')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(Path(output_path).with_suffix('.pdf'), format='pdf')
    plt.savefig(Path(output_path).with_suffix('.eps'), format='eps')
    plt.close()
    print(f"Saved error and forces plot to {output_path}")
    return output_path


def plot_forces_timeseries_comparison(y_true, y_pred, output_path,
                                      f_min=0.1, f_max=0.4, limits=[0.182, 0.19, 0.392, 0.40]):
    """
    Creates a time series plot comparing true and predicted force coefficients.
    If 'limits' are provided, it creates a broken-axis plot focusing on two regions.

    Args:
        y_true (np.ndarray): Array of true values (n_samples, 2).
        y_pred (np.ndarray): Array of predicted values (n_samples, 2).
        output_path (Path): Path object to save the output plot (without extension).
        f_min (float): Minimum frequency for the x-axis.
        f_max (float): Maximum frequency for the x-axis.
        limits (list[float], optional): A list of four values [x1_start, x1_end, x2_start, x2_end]
                                        to create a broken-axis plot. Defaults to None.
    """
    time_steps = np.arange(len(y_true))
    freqs = f_min + (f_max - f_min) * (time_steps / np.max(time_steps))

    line_width = 0.7

    # --- Logic for Broken-Axis Plot ---
    if limits and len(limits) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(8, 4), gridspec_kw={'wspace': 0.02},
                                sharex='col', sharey='row')

        x1_start, x1_end, x2_start, x2_end = limits

        # Data for Cd and Cl plots
        plot_data = [
            {'data_true': y_true[:, 0], 'data_pred': y_pred[:, 0], 'label_true': '$C_d$', 'label_pred': '$\hat{C}_d$'},
            {'data_true': y_true[:, 1], 'data_pred': y_pred[:, 1], 'label_true': '$C_l$', 'label_pred': '$\hat{C}_l$'}
        ]

        for i, (ax_left, ax_right) in enumerate(axs):
            p_data = plot_data[i]

            # Plot on both left and right axes
            ax_left.plot(freqs, p_data['data_true'], color='k', linestyle='-', linewidth=line_width)
            ax_left.plot(freqs, p_data['data_pred'], color='tab:blue', linestyle='--', linewidth=line_width)
            ax_right.plot(freqs, p_data['data_true'], color='k', linestyle='-', linewidth=line_width, label=p_data['label_true'])
            ax_right.plot(freqs, p_data['data_pred'], color='tab:blue', linestyle='--', linewidth=line_width, label=p_data['label_pred'])

            # --- Automatically set tight y-limits based on visible data ---
            # Create masks for the data within the specified frequency limits
            mask1 = (freqs >= x1_start) & (freqs <= x1_end)
            mask2 = (freqs >= x2_start) & (freqs <= x2_end)
            combined_mask = mask1 | mask2
             # Find min/max across both true and predicted data in the visible ranges
            visible_data = np.concatenate(
            (p_data['data_true'][combined_mask], p_data['data_pred'][combined_mask]))
            min_val, max_val = visible_data.min(), visible_data.max()
            padding = (max_val - min_val) * 0.05  # 5% padding
            ax_left.set_ylim(min_val - padding, max_val + padding)

            # Set the x-axis limits for each segment
            ax_left.set_xlim(x1_start, x1_end)
            ax_right.set_xlim(x2_start, x2_end)

            # Apply aesthetics
            ax_left.set_ylabel(p_data['label_true'])
            ax_left.grid(True, linestyle=':', alpha=0.7)
            ax_right.grid(True, linestyle=':', alpha=0.7)
            ax_right.legend(loc='lower right')

            # Hide spines and ticks for a clean look
            ax_left.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_left.spines['top'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.tick_params(axis='y', which='both', left=False) # Hide y-ticks on the right plot

            # Add break marks
            d = .015  # size of the diagonal lines in axes coordinates
            # Left subplot break marks
            kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False, linewidth=line_width)
            ax_left.plot((1 - d/2, 1 + d/2), (-2*d, +2*d), **kwargs)
            # Right subplot break marks
            kwargs.update(transform=ax_right.transAxes)
            ax_right.plot((-d/2, +d/2), (-2*d, +2*d), **kwargs)

        # Common X-label for the bottom plots
        fig.text(0.5, 0.02, '$f_{chirp}$', ha='center', va='center')
        # Hide x-axis labels on the top row of plots
        axs[0, 0].tick_params(axis='x', labelbottom=False)
        axs[0, 1].tick_params(axis='x', labelbottom=False)

        # Use MaxNLocator to prune overlapping ticks in the middle of the bottom row
        axs[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
        axs[1, 1].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

    # --- Original Logic for a Single Continuous Plot ---
    else:
        fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        axs[0].plot(freqs, y_true[:, 0], color='k', linestyle='-', linewidth=0.8, label='$C_d$')
        axs[0].plot(freqs, y_pred[:, 0], color='tab:blue', linestyle='--', linewidth=1.0, label='$\hat{C}_d$')
        axs[0].set_ylabel('$C_d$')
        axs[0].legend(loc='lower right')
        axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)

        axs[1].plot(freqs, y_true[:, 1], color='k', linestyle='-', linewidth=0.8, label='$C_l$')
        axs[1].plot(freqs, y_pred[:, 1], color='tab:blue', linestyle='--', linewidth=1.0, label='$\hat{C}_l$')
        axs[1].set_ylabel('$C_l$')
        axs[1].legend(loc='lower right')
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[1].set_xlabel('$f_{chirp}$')
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure in multiple formats
    output_path = Path(output_path)  # Ensure it's a Path object
    pad = 0.02  # Padding in inches
    plt.savefig(str(output_path) + '.png', format='png', dpi=300, bbox_inches='tight', pad_inches=pad)
    plt.savefig(str(output_path) + '.pdf', format='pdf', bbox_inches='tight', pad_inches=pad)
    plt.savefig(str(output_path) + '.eps', format='eps', bbox_inches='tight', pad_inches=pad)
    print(f"Force decoder direct evaluation plot saved to {output_path.with_suffix('.png')}")
    plt.close(fig)

    return output_path.with_suffix('.png')



def plot_forces_scatter_comparison(y_true, y_pred, output_path, reverse=True, f_min=0.1, f_max=0.4):
    """
    Creates a side-by-side scatter plot of true vs. predicted force coefficients,
    colored by time step.

    Args:
        y_true (np.ndarray): Array of true values (n_samples, 2).
        y_pred (np.ndarray): Array of predicted values (n_samples, 2).
        output_path (Path): Path object to save the output plot (without extension).
    """
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 2:
        print("Error: Input arrays for scatter plot must have shape (n_samples, 2).")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    time_steps = np.arange(len(y_true))

    # Change points order
    if reverse:
        y_true = y_true[::-1]
        y_pred = y_pred[::-1]
        time_steps = time_steps[::-1]

    freqs = f_min + (f_max-f_min)*(time_steps/np.max(time_steps))

    # --- Cd Scatter Plot (Left) ---
    scatter = ax1.scatter(y_true[:, 0], y_pred[:, 0], c=freqs, cmap='viridis', s=1, alpha=0.7, rasterized=True)
    ax1.set_xlabel('Reference $C_d$')
    ax1.set_ylabel('Predicted $\hat{C}_d$')

    # Add identity line
    lims1 = [
        min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
        max(ax1.get_xlim()[1], ax1.get_ylim()[1]),
    ]
    ax1.plot(lims1, lims1, 'k--', alpha=0.75, zorder=0, linewidth=1)
    ax1.set_xlim(lims1)
    ax1.set_ylim(lims1)
    ax1.set_aspect('equal', adjustable='box')

    # Aesthetics
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Cl Scatter Plot (Right) ---
    ax2.scatter(y_true[:, 1], y_pred[:, 1], c=freqs, cmap='viridis', s=1, alpha=0.7, rasterized=True)
    ax2.set_xlabel('Reference $C_l$')
    ax2.set_ylabel('Predicted $\hat{C}_l$')

    # Add identity line
    lims2 = [
        min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
        max(ax2.get_xlim()[1], ax2.get_ylim()[1]),
    ]
    ax2.plot(lims2, lims2, 'k--', alpha=0.75, zorder=0, linewidth=1)
    ax2.set_xlim(lims2)
    ax2.set_ylim(lims2)
    ax2.set_aspect('equal', adjustable='box')

    # Aesthetics
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Add a shared colorbar and adjust layout
    fig.tight_layout()
    fig.subplots_adjust(right=0.85) # Fine-tune space for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('$f_{chirp}$')
    cbar.outline.set_visible(False)

    # Save the figure
    plt.savefig(output_path.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.with_suffix('.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Force decoder scatter plot saved to {output_path.with_suffix('.png')}")

    return output_path


def find_model_paths(model_strdate_identifier, checkpoints_base_dir):
    """
    Finds the corresponding checkpoint directories and MLflow run directories based on the model identifier.

    Args:
        model_strdate_identifier (str): The strdate identifier of the model (e.g., '20250301_16_29_06').
        checkpoints_base_dir (str): Base directory for checkpoints.

    Returns:
        Path | None: Containing the checkpoint directory, or None if not found.
        Path | None: Containing the results directory, or None if not found.
    """
    checkpoints_base_path = Path(checkpoints_base_dir)

    matching_case_dirs = []
    for case_dir in checkpoints_base_path.iterdir():
        if case_dir.is_dir():
            for item in case_dir.iterdir():
                if model_strdate_identifier in item.name:
                    matching_case_dirs.append(case_dir)
                    break

    if not matching_case_dirs:
        print(f"No case directories found for model identifier: {model_strdate_identifier}")
        return None, None
    elif len(matching_case_dirs) > 1:
        print(
            f"More than one case directory found for model identifier: {model_strdate_identifier}, taking the first one")
    else:
        pass

    case_dir = matching_case_dirs[0]

    if not case_dir.exists():
        print(f"Checkpoint directory not found: {case_dir}")
        return None, None

    results_dir = Path(str(case_dir).replace('03_Checkpoints', '04_Results'))

    return case_dir, results_dir


def find_checkpoint_files(checkpoint_dir, model_identifier):
    """
    Finds the encoder and dynamics checkpoint files in the given directory based on the model identifier.

    Args:
        checkpoint_dir (Path): The directory containing the checkpoint files.
        model_identifier (str): The identifier string to match in the filenames.

    Returns:
        tuple: Tuple containing the paths to the encoder and dynamics checkpoint files, or (None, None) if not found.
    """
    encoder_ckpt_pattern = re.compile(rf"{model_identifier}.*_encoder\.pth\.tar")
    dynamics_ckpt_pattern = re.compile(rf"{model_identifier}.*_dynamics\.pth\.tar")
    forces_decoder_ckpt_pattern = re.compile(rf"{model_identifier}.*_force\.pth\.tar")

    encoder_ckpt_path = next((f for f in checkpoint_dir.iterdir() if encoder_ckpt_pattern.match(f.name)), None)
    dynamics_ckpt_path = next((f for f in checkpoint_dir.iterdir() if dynamics_ckpt_pattern.match(f.name)), None)

    if encoder_ckpt_path is None or dynamics_ckpt_path is None:
        print(f"Could not find both encoder and dynamics checkpoint files in {checkpoint_dir}")
        return None, None
    forces_decoder_ckpt_path = next((f for f in checkpoint_dir.iterdir() if forces_decoder_ckpt_pattern.match(f.name)), None)

    return encoder_ckpt_path, dynamics_ckpt_path, forces_decoder_ckpt_path


def compare_recursive_latent_evolution(true_latents, control_seq, idx_init, horizon, dynamics_model, device):
    """
    Compare the true latent evolution (already encoded) with the recursively predicted latent evolution.

    Args:
        true_latents (torch.Tensor): Tensor of true latent states, shape (horizon+1, latent_dim).
                                     These are the encoded states at each time step.
        control_seq (torch.Tensor): Tensor of control actions for each prediction step, shape (horizon, action_dim).
        idx_init (int): Index to start prediction
        horizon (int): Prediction horizon (number of recursive steps).
        dynamics_model (nn.Module): The trained latent dynamics model that maps (latent state, control) to next latent state.
        device (torch.device): Device on which to run the computations (e.g., torch.device("cuda") or torch.device("cpu")).

    Returns:
        true_latents_np (np.ndarray): True latent states as a NumPy array, shape (horizon+1, latent_dim).
        predicted_latents_np (np.ndarray): Recursively predicted latent states as a NumPy array, shape (horizon+1, latent_dim).
        error_series (np.ndarray): Mean absolute error (averaged over latent dimensions) at each time step, shape (horizon+1,).
    """
    dynamics_model.eval()

    # Ensure the inputs are on the correct device.
    true_latents = true_latents.to(device)
    control_seq = control_seq.to(device)

    # List to collect predicted latent states.
    predicted_latents = []

    # The initial predicted latent state equals the true initial latent.
    z_pred = true_latents[idx_init]
    predicted_latents.append(z_pred)

    # Recursively predict latent states using the dynamics model.
    with torch.no_grad():
        for t in range(horizon):
            u_t = control_seq[t].unsqueeze(0)  # shape: (1, action_dim)
            # Predict the next latent state (dynamics_model expects input shape (batch, latent_dim) for z and (batch, action_dim) for u)
            z_pred = dynamics_model(z_pred.unsqueeze(0), u_t).squeeze(0)
            predicted_latents.append(z_pred)

    # Convert lists to tensors and then to NumPy arrays.
    predicted_latents = torch.stack(predicted_latents, dim=0)  # shape: (horizon+1, latent_dim)
    true_latents_np = true_latents.cpu().numpy()[idx_init:idx_init+horizon+1]
    predicted_latents_np = predicted_latents.cpu().numpy()

    # Compute the mean absolute error over latent dimensions at each time step.
    error_series = np.mean(np.abs(true_latents_np - predicted_latents_np), axis=1)

    return true_latents_np, predicted_latents_np, error_series


def test_encoder_predictor(model_strdate_identifier, use_cuda=True, checkpoints_base_dir="03_Checkpoints"):
    """Test the encoder and predictor models."""
    # ... (initial setup, finding paths, loading training args/scaling remains the same) ...
    print(f"\n--- Starting Evaluation for Model ID: {model_strdate_identifier} ---")

    # --- Setup ---
    case_ckp_dir, case_results_dir = find_model_paths(model_strdate_identifier, checkpoints_base_dir)
    if case_ckp_dir is None or case_results_dir is None:
        print("Could not find the required case directories. Exiting.")
        return None, None, None, None, None

    results_subdir = case_results_dir / model_strdate_identifier
    results_subdir.mkdir(parents=True, exist_ok=True)
    print(f'Results will be saved in: {results_subdir}')

    # --- Load Training Configuration and Scaling ---
    latent_file_pattern = f"{model_strdate_identifier}*latent_space.hdf5"
    latent_files = sorted(list(case_results_dir.glob(latent_file_pattern)))
    latent_file = next((f for f in latent_files if '_eval' not in f.name), None)
    if latent_file is None and latent_files:
        latent_file = latent_files[0]
        print(f"Warning: Using training latent file with potential suffix: {latent_file.name}")
    elif not latent_files:
         print(f"Error: No training latent file matching pattern '{latent_file_pattern}' found in {case_results_dir}")
         return None, None, None, None, None
    latent_file = str(latent_file)
    print(f'Loading training config and scaling from: {latent_file}')
    try:
        (_, _, forces_mean_train, forces_std_train, _, _, _, args_loaded) = load_latents_file(latent_file, printer=True)
        if not all([hasattr(args_loaded, attr) for attr in ['eval_dataset', 'lookback', 'recursive_train_steps']]):
             raise ValueError("Loaded args missing required attributes (eval_dataset, lookback, recursive_train_steps).")
    except Exception as e:
        print(f"Error loading training config/scaling from {latent_file}: {e}")
        return None, None, None, None, None

    # --- Prepare Evaluation Data ---
    project_root = Path(checkpoints_base_dir).parent
    # eval_dataset can now be a list of strings or a single string
    eval_dataset_paths_raw = args_loaded.eval_dataset
    if isinstance(eval_dataset_paths_raw, str):
        eval_dataset_paths_raw = [eval_dataset_paths_raw]  # Ensure it's a list

    eval_dataset_paths = [str(project_root / p) for p in eval_dataset_paths_raw]
    print(f'Using evaluation dataset(s): {eval_dataset_paths}')

    # Check if all eval files exist
    for p in eval_dataset_paths:
        if not Path(p).exists():
            print(f"Error: Evaluation dataset not found at {p}")
            return None, None, None, None, None

    # --- Define batch_size BEFORE using it in the class ---
    eval_batch_size = args_loaded.batch_size if hasattr(args_loaded, 'batch_size') else 256 # Use loaded or default

    # --- <<< CRITICAL CHANGE for EVALUATION DATALOADER >>> ---
    # Use recursive_train_steps = 1 for evaluation loading, regardless of training value.
    # This ensures a_seq and c_seq have lookforward=1 for simpler step-by-step evaluation.
    class EvalLoaderArgs:
        datafile = eval_dataset_paths # Pass the list of full paths
        n_test = 0 # Use all data from eval files for evaluation
        lookback = args_loaded.lookback
        recursive_train_steps = 1 # Force lookforward=1 for step-by-step evaluation
        batch_size = eval_batch_size
        augment_with_symmetry = False # Never augment for evaluation
        DATA_TO_GPU = False
        sel_coefs = ['Cd', 'Cl'] # Assuming this is standard

    print(f"Preparing evaluation dataloader with lookforward = {EvalLoaderArgs.recursive_train_steps}, batch_size = {EvalLoaderArgs.batch_size}")
    try:
        dataloader_eval, _ = get_prepared_data(EvalLoaderArgs, device=torch.device('cpu'), shuffle_train=False, shuffle_test=False)
    except Exception as e:
         print(f"Error creating evaluation dataloader: {e}")
         return None, None, None, None, None
    if len(dataloader_eval.dataset) == 0:
         print(f"Error: Evaluation dataloader is empty.")
         return None, None, None, None, None
    print(f"Loaded evaluation dataset: {len(dataloader_eval.dataset)} samples.")

    # Load raw forces from the evaluation dataset for plotting context
    try:
        # Create a temporary args-like object for loadData
        class TempLoadArgs:
            datafile = eval_dataset_paths
            augment_with_symmetry = False # Never augment for evaluation
        (_, forces_unscaled_eval, _, _, _, _, _, _, _, _, _) = loadData(TempLoadArgs, sel_coefs=['Cd', 'Cl'])
        # Unscale using TRAINING mean/std for consistency
        forces_eval_unscaled = forces_unscaled_eval
        print(f"Loaded evaluation forces. Shape: {forces_eval_unscaled.shape}")
    except Exception as e:
        print(f"Warning: Could not load forces from eval dataset for plotting: {e}")
        forces_eval_unscaled = None

    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"Using device: {device}")

    # --- Load Models ---
    model_id_for_ckpts = args_loaded.modelname if hasattr(args_loaded, 'modelname') and args_loaded.modelname else model_strdate_identifier
    print(f"Searching for checkpoints in {case_ckp_dir} using identifier: {model_id_for_ckpts}")
    try:
        encoder_ckpt, dynamics_ckpt, forces_ckpt = find_checkpoint_files(case_ckp_dir, model_id_for_ckpts)
        if not forces_ckpt: print("Force decoder checkpoint not found, force evaluation will be skipped.")
    except FileNotFoundError as e:
        print(f"Error finding essential checkpoints: {e}")
        return None, None, None, None, None

    # Initialize models
    try:
        sample_s_t, sample_a_seq, _, _ = dataloader_eval.dataset[0]
        in_dim = sample_s_t.shape[-1]
        action_dim = sample_a_seq.shape[-1] # Should be 1 now due to lookforward=1
    except Exception as e:
        print(f"Error getting dimensions from dataloader sample: {e}. Exiting.")
        return None, None, None, None, None

    print(f"Initializing models: input_dim={in_dim}, action_dim={action_dim}, latent_dim={args_loaded.latent_dim}")
    encoder = TemporalEncoder(input_dim=in_dim, latent_dim=args_loaded.latent_dim, hidden_dim=args_loaded.enc_hidden_dim, num_layers=args_loaded.n_layers).to(device)
    # Ensure use_residual is correctly read from args_loaded
    use_residual_flag = args_loaded.residual_predictor if hasattr(args_loaded, 'residual_predictor') else False # Default to False if not in args
    dynamics_model = LatentDynamicsModel(latent_dim=args_loaded.latent_dim, action_dim=action_dim, hidden_dim=args_loaded.dyn_hidden_dim, use_residual=use_residual_flag).to(device)
    forces_decoder = ForceDecoder(latent_dim=args_loaded.latent_dim, hidden_dim=args_loaded.forces_hidden_dim,
                                  arch=args_loaded.force_decoder_arch).to(device) if forces_ckpt else None

    # Load state dicts
    try:
        load_checkpoint(encoder, encoder_ckpt, device)
        load_checkpoint(dynamics_model, dynamics_ckpt, device)
        if forces_decoder and forces_ckpt:
            load_checkpoint(forces_decoder, forces_ckpt, device)
    except (FileNotFoundError, RuntimeError) as e:
         print(f"Error loading checkpoint state dict: {e}") # Corrected return statement
         return None, None, None, None, None

    # --- Evaluate Latent Prediction (Single Step) ---
    print("\nEvaluating single-step latent prediction error...")
    # Pass action_dim needed by evaluate_models_over_time
    errors_over_time, indices = evaluate_models_over_time(encoder, dynamics_model, forces_decoder, dataloader_eval, device)

    error_plot_path = None, None
    if len(errors_over_time) > 0:
        mse_mean = np.mean(errors_over_time)
        mse_std = np.std(errors_over_time)
        print(f"Mean latent prediction standardized MSE: {mse_mean:.4f} � {mse_std:.4f}")

        output_plot_path_time = results_subdir / "latent_error_over_time_eval.png"
        # Align forces: Error at index `j` corresponds to prediction for time `j + lookback`.
        offset = EvalLoaderArgs.lookback # Use the lookback from EvalLoaderArgs
        aligned_forces = None
        if forces_eval_unscaled is not None:
             expected_len = len(indices)
             if len(forces_eval_unscaled) >= offset + expected_len:
                 aligned_forces = forces_eval_unscaled[offset : offset + expected_len]
                 if len(aligned_forces) != expected_len:
                      print(f"Warning: Sliced forces length {len(aligned_forces)} != indices length {expected_len}.")
                      aligned_forces = None
             else:
                 print(f"Warning: Not enough force data (len {len(forces_eval_unscaled)}) to align with indices (need up to {offset + expected_len}).")
        error_plot_path = plot_error_over_time(errors_over_time, indices, output_plot_path_time)

    else:
        print("Skipping latent error plotting due to zero errors calculated.")


    # --- Evaluate Recursive Latent Prediction ---
    print("\nEvaluating recursive latent prediction by encoding eval dataset...")
    pred_sample_plot_path = None
    horizon = 50
    idx_start_pred = 100

    eval_latents_list = []
    eval_controls_list = []
    encoder.eval()
    print("Encoding evaluation dataset for recursive test...")
    with torch.no_grad():
        for s_t, a_seq, _, _ in dataloader_eval:
            z_t = encoder(s_t.to(device))
            eval_latents_list.append(z_t.cpu())
            eval_controls_list.append(a_seq[:, 0, :].cpu()) # Control for next step
    if not eval_latents_list:
         print("Error: Could not generate latent states from eval dataloader. Skipping recursive plot.")
    else:
        full_eval_latent_space = torch.cat(eval_latents_list, dim=0)
        full_eval_control_sequence = torch.cat(eval_controls_list, dim=0)
        print(f"Generated eval latents: {full_eval_latent_space.shape}, eval controls: {full_eval_control_sequence.shape}")
        try:
            true_latents_np, predicted_latents_np, _ = compare_recursive_latent_evolution(
                full_eval_latent_space, full_eval_control_sequence, idx_start_pred, horizon, dynamics_model, device
            )
            if true_latents_np.size > 0:
                 output_plot_path_pred = results_subdir / "pred_sample_eval.png"
                 pred_sample_plot_path = plot_pred_sample(true_latents_np, predicted_latents_np, output_plot_path_pred)
            else:
                 print("Recursive comparison returned empty arrays. Skipping plot.")
        except (ValueError, IndexError, TypeError) as e: # Catch more potential errors
             print(f"Error during recursive prediction/plotting: {e}")


    # --- Evaluate Forces Decoder ---
    print("\nEvaluating forces decoder error...")
    forces_error_plot_path = None
    if forces_decoder and forces_ckpt:
        # Pass TRAINING mean/std for unscaling
        forces_errors_over_time, forces_indices = evaluate_forces_decoder(encoder, forces_decoder, dataloader_eval, device, forces_mean_train, forces_std_train)

        if len(forces_errors_over_time) > 0:
            forces_mae_mean_cd = np.mean(forces_errors_over_time[:, 0])
            forces_mae_std_cd = np.std(forces_errors_over_time[:, 0])
            forces_mae_mean_cl = np.mean(forces_errors_over_time[:, 1])
            forces_mae_std_cl = np.std(forces_errors_over_time[:, 1])
            print(f"Mean forces prediction Abs Error: Cd={forces_mae_mean_cd:.4f}�{forces_mae_std_cd:.4f}, Cl={forces_mae_mean_cl:.4f}�{forces_mae_std_cl:.4f}")

            output_plot_path_forces_time = results_subdir / "forces_error_over_time_eval.png"
            # Align forces: Force target c_t corresponds to time j + lookback + recursive_steps(=1) - 1 = j + lookback
            force_offset = EvalLoaderArgs.lookback # Offset is just lookback when lookforward=1
            aligned_forces_for_force_plot = None
            if forces_eval_unscaled is not None:
                 expected_len = len(forces_indices)
                 if len(forces_eval_unscaled) >= force_offset + expected_len:
                     aligned_forces_for_force_plot = forces_eval_unscaled[force_offset : force_offset + expected_len]
                     if len(aligned_forces_for_force_plot) != expected_len:
                          print(f"Warning: Sliced forces length {len(aligned_forces_for_force_plot)} != indices length {expected_len} for force error plot.")
                          aligned_forces_for_force_plot = None
                 else:
                      print(f"Warning: Not enough force data (len {len(forces_eval_unscaled)}) to align with indices (need up to {force_offset + expected_len}) for force error plot.")
            forces_error_plot_path = plot_forces_error_over_time(forces_errors_over_time, forces_indices, output_plot_path_forces_time, aligned_forces_for_force_plot)
        else:
            print("Skipping forces error plotting due to zero errors calculated.")
    else:
        print("Skipping force evaluation as decoder model/checkpoint was not loaded.")

    # --- Evaluate Pre-trained ForceDecoder Directly ---
    print("\nEvaluating pre-trained ForceDecoder performance directly on eval latents...")
    decoder_eval_plot_path = None  # Initialize path variable
    scatter_plot_path = None # Initialize path variable for the new plot

    # Check if necessary components are available
    if forces_decoder and forces_ckpt and 'full_eval_latent_space' in locals() and full_eval_latent_space is not None:
        forces_decoder.eval()  # Set to eval mode

        y_pred_scaled_list = []
        eval_pred_batch_size = 1024  # Batch size for prediction (adjust if needed)

        # Create a simple dataset/loader for prediction
        temp_dataset = TensorDataset(full_eval_latent_space)  # Use the generated latents
        temp_loader = DataLoader(temp_dataset, batch_size=eval_pred_batch_size)

        print(f"Predicting forces for {len(full_eval_latent_space)} eval samples...")
        with torch.no_grad():
            for (x_batch,) in temp_loader:
                preds = forces_decoder(x_batch.to(device))
                y_pred_scaled_list.append(preds.cpu())

        y_pred_scaled = torch.cat(y_pred_scaled_list).numpy()

        # Unscale predictions using TRAINING mean/std
        forces_mean_np = np.array(forces_mean_train)
        forces_std_np = np.array(forces_std_train)
        y_pred_unscaled = y_pred_scaled * forces_std_np + forces_mean_np
        print("Unscaled ForceDecoder predictions.")

        # Align true forces: z_t for sample j corresponds to time j + lookback - 1
        # We need forces_eval_unscaled[j + lookback - 1]
        force_offset_for_decoder = EvalLoaderArgs.lookback - 1
        y_true_unscaled = None
        decoder_results = None

        if forces_eval_unscaled is not None:
            expected_len = len(full_eval_latent_space)
            if len(forces_eval_unscaled) >= force_offset_for_decoder + expected_len:
                y_true_unscaled = forces_eval_unscaled[
                                  force_offset_for_decoder: force_offset_for_decoder + expected_len]
                # Final check on length after slicing
                if len(y_true_unscaled) != expected_len:
                    print(
                        f"Warning: Alignment logic error for true forces. Expected {expected_len}, got {len(y_true_unscaled)}. Skipping metrics/plot.")
                    y_true_unscaled = None
                else:
                    print("Aligned true forces for direct decoder evaluation.")
            else:
                print(
                    f"Warning: Not enough force data (len {len(forces_eval_unscaled)}) to align for decoder evaluation (need {force_offset_for_decoder + expected_len}).")
        else:
            print("Warning: Raw evaluation forces not available for direct decoder evaluation.")

        # Calculate metrics if true forces are available and aligned
        if y_true_unscaled is not None:
            print("Calculating direct ForceDecoder metrics...")
            mae_decoder = mean_absolute_error(y_true_unscaled, y_pred_unscaled, multioutput='raw_values')
            mse_decoder = mean_squared_error(y_true_unscaled, y_pred_unscaled, multioutput='raw_values')
            r2_decoder = r2_score(y_true_unscaled, y_pred_unscaled, multioutput='raw_values')
            print(f"  Direct Eval MAE (Cd, Cl): {mae_decoder}")
            print(f"  Direct Eval MSE (Cd, Cl): {mse_decoder}")
            print(f"  Direct Eval R2  (Cd, Cl): {r2_decoder}")
            decoder_results = {'MAE': mae_decoder.tolist(), 'MSE': mse_decoder.tolist(), 'R2': r2_decoder.tolist()}

            # Plotting time series comparison
            print("Plotting direct ForceDecoder time series comparison...")
            decoder_eval_plot_path_base = results_subdir / (
            f'force_eval_CdMAE{decoder_results["MAE"][0]:.4f}'
            f'_CdR2_{decoder_results["R2"][0]:.3f}'
            f'_ClMAE{decoder_results["MAE"][1]:.4f}'
            f'_ClR2_{decoder_results["R2"][1]:.3f}')
            decoder_eval_plot_path = plot_forces_timeseries_comparison(y_true_unscaled, y_pred_unscaled,
                                                                       decoder_eval_plot_path_base)


            # Plotting scatter comparison
            print("Plotting direct ForceDecoder scatter comparison...")
            scatter_plot_path_base = results_subdir / (f'force_eval_scatter_CdR2_{decoder_results["R2"][0]:.3f}'
                                                       f'_ClR2_{decoder_results["R2"][1]:.3f}.png')
            scatter_plot_path = plot_forces_scatter_comparison(y_true_unscaled, y_pred_unscaled, scatter_plot_path_base)
        else:
            print(
                "Skipping direct force decoder metrics calculation and plotting due to missing/mismatched true forces.")

    else:
        if not (forces_decoder and forces_ckpt):
            print("Skipping direct ForceDecoder evaluation: Decoder/checkpoint not loaded.")
        elif not ('full_eval_latent_space' in locals() and full_eval_latent_space is not None):
            print("Skipping direct ForceDecoder evaluation: Evaluation latent space not available.")

    plt.close('all')
    print(f"\n--- Evaluation Finished for Model ID: {model_strdate_identifier} ---")
    print(f"Results saved in: {results_subdir}")

    return error_plot_path, forces_error_plot_path, pred_sample_plot_path, decoder_eval_plot_path, scatter_plot_path


if __name__ == "__main__":

    str_id = '20250906_19_38_58'

    test_encoder_predictor(str_id, checkpoints_base_dir="../03_Checkpoints")