import matplotlib
matplotlib.use('Agg')  # Set the backend BEFORE importing pyplot
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm
import subprocess # Already there
from functools import partial # Add this for passing arguments
from multiprocessing import Pool # Add this for parallelization
import os # Add this to get CPU count
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def load_mpc_data(hdf5_path):
    """Loads all necessary data from the MPC results HDF5 file."""
    data = {}
    with h5py.File(hdf5_path, 'r') as f:
        data['actual_controls'] = f['actions_applied_unscaled'][:]
        data['actual_forces'] = f['actual_forces'][:]
        data['predicted_controls'] = f['predicted_control_sequences'][:]
        data['predicted_forces'] = f['predicted_force_sequences'][:]

        # Infer parameters
        data['num_steps'] = data['actual_controls'].shape[0]
        data['horizon'] = data['predicted_controls'].shape[1]
        data['control_dt'] = 0.2  # As configured in MPC_onFOM.py
        data['time_axis'] = np.arange(data['num_steps']) * data['control_dt']
        data['control_max'] = 0.075 #max(abs(data['actual_controls']))

    print("Data loaded successfully.")
    print(f"  Total Steps: {data['num_steps']}")
    print(f"  Horizon: {data['horizon']}")
    return data


def create_frame(t_idx, data, output_dir, title=True, custom_filename=None):
    """Creates a single plot frame, can be a video frame or a special final summary image."""

    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    # Create a 1x2 grid: left side for subplots, right side for colorbar
    gs = fig.add_gridspec(1, 2, width_ratios=[60, 1], wspace=0.05)
    # Create a nested GridSpec for the 3 subplots on the left
    gs_plots = gs[0].subgridspec(3, 1)

    axs = gs_plots.subplots(sharex=True)  # Create the 3 axes
    cbar_ax = fig.add_subplot(gs[1])  # Create the single axis for the colorbar

    if title:
        fig.suptitle(f'MPC Predictions at Time: {data["time_axis"][t_idx]:.2f} s', fontsize=16)

    time_axis = data['time_axis']
    horizon = data['horizon']

    cmap = plt.get_cmap('summer')
    # Normalize horizon steps (from 0 to horizon-1) to the [0, 1] range for the colormap
    norm = mcolors.Normalize(vmin=0, vmax=horizon)

    # --- 1. Plot all past predictions ---
    for past_t in range(t_idx):
        past_time_pred_axis = time_axis[past_t:past_t + horizon]
        past_control_preds = data['predicted_controls'][past_t] / data['control_max']
        past_forces_preds = data['predicted_forces'][past_t]

        for h in range(horizon):
            # --- MODIFICATION: Use the normalizer for color ---
            color = cmap(norm(h))
            if h < len(past_time_pred_axis):
                # Plot as small points so they don't connect
                axs[0].plot(past_time_pred_axis[h], past_control_preds[h, 0], 'o', color=color, markersize=1, rasterized=True)
                axs[1].plot(past_time_pred_axis[h], past_forces_preds[h, 0],  'o', color=color, markersize=1, rasterized=True)  # Cd
                axs[2].plot(past_time_pred_axis[h], past_forces_preds[h, 1],  'o', color=color, markersize=1, rasterized=True)  # Cl

    # --- 2. Plot the CURRENT prediction horizon ---
    current_time_pred_axis = time_axis[t_idx:t_idx + horizon]
    current_control_preds = data['predicted_controls'][t_idx] / data['control_max']
    current_forces_preds = data['predicted_forces'][t_idx]

    for h in range(horizon):
        color = cmap(norm(h))
        if h < len(current_time_pred_axis):
            # Use slightly larger markers and higher alpha for the current prediction
            axs[0].plot(current_time_pred_axis[h], current_control_preds[h, 0], 'o', color=color, markersize=2, rasterized=True)
            axs[1].plot(current_time_pred_axis[h], current_forces_preds[h, 0],  'o', color=color, markersize=2, rasterized=True)
            axs[2].plot(current_time_pred_axis[h], current_forces_preds[h, 1],  'o', color=color, markersize=2, rasterized=True)

    # --- 3. Plot the ground truth solid line ---
    time_so_far = time_axis[:t_idx + 1]
    actual_controls_so_far = data['actual_controls'][:t_idx + 1] / data['control_max']
    actual_forces_so_far = data['actual_forces'][:t_idx + 1]

    axs[0].plot(time_so_far, actual_controls_so_far, 'k-', linewidth=2, label='Actual control')
    axs[1].plot(time_so_far, actual_forces_so_far[:, 0], 'k-', linewidth=2, label='Actual $C_d$')
    axs[2].plot(time_so_far, actual_forces_so_far[:, 1], 'k-', linewidth=2, label='Actual $C_l$')

    # --- Formatting ---
    total_duration = (time_axis[-1]//10 + 1) * 10
    axs[0].set_ylabel('Control Action')
    axs[1].set_ylabel(f'$C_d$')
    axs[2].set_ylabel(f'$C_l$')
    axs[2].set_xlabel(f'$t_c$')

    # We use the color for the middle of the horizon as a representative color
    representative_color = cmap(norm(horizon / 2))
    for ax in axs:
        # Plot an invisible point to create the legend handle
        ax.plot([], [], 'o', color=representative_color, markersize=5, label='Predicted')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1, 1.06), frameon=False, columnspacing=1, handletextpad=0.5)
        ax.set_xlim(0, total_duration)

    # Set axis limits ---
    # Control Action (axs[0])
    # Find max absolute value from both actual and predicted data
    max_abs_ctrl = 1
    max_abs_pred_ctrl = np.nanmax(np.abs(data['predicted_controls']))
    # Use the larger of the two to set the limits, add some padding
    final_max_ctrl = max(max_abs_ctrl, max_abs_pred_ctrl)
    if np.isfinite(final_max_ctrl) and final_max_ctrl > 0:
        axs[0].set_ylim(-final_max_ctrl, final_max_ctrl)

    # Drag Coefficient (axs[1])
    # Find overall min/max from both actual and predicted data
    min_val_cd = np.nanmin([np.nanmin(data['actual_forces'][:, 0]), np.nanmin(data['predicted_forces'][:, :, 0])])
    max_val_cd = np.nanmax([np.nanmax(data['actual_forces'][:, 0]), np.nanmax(data['predicted_forces'][:, :, 0])])

    # Calculate limits rounded to the next multiple of 0.05
    multiple = 0.05
    if np.isfinite(min_val_cd) and np.isfinite(max_val_cd):
        # Floor the min limit, Ceil the max limit
        new_min_cd = np.floor(min_val_cd / multiple) * multiple
        new_max_cd = np.ceil(max_val_cd / multiple) * multiple

        # Add a small padding if min and max are too close
        if np.isclose(new_min_cd, new_max_cd):
            new_max_cd += multiple

        axs[1].set_ylim(new_min_cd, new_max_cd)

        # Set y-ticks to be multiples of 0.05 within the new limits
        # Use a small epsilon to ensure the upper limit is included in arange
        yticks = np.arange(new_min_cd, new_max_cd + (multiple * 0.1), multiple)
        axs[1].set_yticks(yticks)

    # Lift Coefficient (axs[2])
    # Find max absolute value from both actual and predicted data
    max_abs_cl = np.nanmax(np.abs(data['actual_forces'][:, 1]))
    max_abs_pred_cl = np.nanmax(np.abs(data['predicted_forces'][:, :, 1]))
    # Use the larger of the two to set the limits, add some padding
    final_max_cl = max(max_abs_cl, max_abs_pred_cl) * 1.02
    if np.isfinite(final_max_cl) and final_max_cl > 0:
        axs[2].set_ylim(-final_max_cl, final_max_cl)

    # Create a dummy ScalarMappable object that has the colormap and normalizer
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # Draw the colorbar in the dedicated axis
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # Generate ~5 ticks, ensuring 1 and horizon are included
    num_ticks = 5
    ticks = np.linspace(0, horizon, num=num_ticks, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_label('Prediction horizon step')

    if custom_filename:
        # Save the special static image in the parent directory (alongside the HDF5 file)
        save_path = output_dir.parent / custom_filename
        plt.savefig(save_path.with_suffix('.png'), dpi=300)
        plt.savefig(save_path.with_suffix('.pdf'))
        plt.savefig(save_path.with_suffix('.eps'))

    else:
        # Regular frame for the video, saved inside the frames directory
        save_path = output_dir / f"frame_{t_idx:05d}.png"
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def main(input_file, final_plot_only=False, paralell_proc=True):
    hdf5_path = Path(input_file)
    if not hdf5_path.is_file():
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return

    # Create output directory for frames, named after the input file + "_frames"
    output_path = hdf5_path.parent / "video_frames"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Frames will be saved in: {output_path}")

    mpc_data = load_mpc_data(hdf5_path)


    complete_plot_name = f"mpc_predictions_{hdf5_path.stem}"  # Without extension
    create_frame(mpc_data['num_steps'] - 1, mpc_data, output_path,
                 title=False, custom_filename=complete_plot_name)
    print(f"Saved complete plot to: {complete_plot_name}")

    if not final_plot_only:
        if paralell_proc:

            # Use less than the total number of cores to leave resources for the OS
            # Or set to a specific number, e.g., n_workers = 8
            n_workers = max(1, os.cpu_count()//2)
            print(f"Using {n_workers} worker processes for frame generation.")

            # The create_frame function needs multiple arguments (t_idx, data, output_dir).
            # We use functools.partial to "pre-fill" the arguments that don't change,
            # leaving only the first argument (t_idx) to be supplied by the pool.
            task_func = partial(create_frame, data=mpc_data, output_dir=output_path)

            # Create a pool of worker processes
            with Pool(processes=n_workers) as pool:
                # Use pool.imap_unordered for efficiency. It processes items as they are submitted
                # and returns results as they complete, which works perfectly with tqdm.
                # We wrap the pool iterator with tqdm to create a progress bar.
                list(tqdm(pool.imap_unordered(task_func, range(mpc_data['num_steps'])),
                          total=mpc_data['num_steps'],
                          desc="Generating Frames"))

        else:
            for t in tqdm(range(mpc_data['num_steps']), desc="Generating Frames"):
                create_frame(t, mpc_data, output_path)

        print("\nFrame generation complete.")

        # Define paths for FFmpeg command
        input_frames_pattern_abs = str((output_path / 'frame_%05d.png').resolve())
        output_video_name = f"mpc_predictions_{hdf5_path.stem}.mp4"
        # Save the video in the same directory as the HDF5 file
        output_video_path_abs = str((hdf5_path.parent / output_video_name).resolve())

        # Build the command as a list for subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-r', '15', '-i', input_frames_pattern_abs,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', output_video_path_abs
        ]

        # Print a user-friendly version of the command
        print("You can create a video using a tool like FFmpeg. Trying to launch it automatically:")
        print(
            f"ffmpeg -r 15 -i \"{output_path}/frame_%05d.png\" -c:v libx264 -pix_fmt yuv420p -y \"{output_video_path_abs}\"")

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"\nVideo successfully created: {output_video_path_abs}")
        except FileNotFoundError:
            print("\nError: 'ffmpeg' command not found. Please install FFmpeg and run the command above manually.")
        except subprocess.CalledProcessError as e:
            print("\nError during video creation with FFmpeg.")
            print(f"FFmpeg Stderr: {e.stderr}")


if __name__ == "__main__":
    # --- USER SETTING: Set to True to only generate the final summary plot ---
    FINAL_PLOT_ONLY = True

    MPClogfile = '../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5'

    main(input_file=MPClogfile, final_plot_only=FINAL_PLOT_ONLY, paralell_proc=True)
