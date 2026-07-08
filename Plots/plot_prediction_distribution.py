import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})


def create_error_distribution_plot(input_file):
    """
    Loads MPC results data and creates a plot showing the mean prediction
    error and the standard deviation over the prediction horizon.
    """
    hdf5_path = Path(input_file)
    if not hdf5_path.is_file():
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return

    # --- 1. Load Data ---
    with h5py.File(hdf5_path, 'r') as f:
        actual_forces = f['actual_forces'][:]
        predicted_forces = f['predicted_force_sequences'][:]

    num_steps, horizon, num_force_coeffs = predicted_forces.shape
    print(f"Data loaded from {hdf5_path.name}")
    print(f"actual_forces {actual_forces.shape}")
    print(f"predicted_forces {predicted_forces.shape}")

    print(f"  - Number of time steps: {num_steps}")
    print(f"  - Prediction horizon (H): {horizon}")

    # --- 2. Calculate Prediction Errors ---
    num_eval_steps = num_steps - horizon
    errors = np.zeros((num_eval_steps, horizon, num_force_coeffs))

    for h in range(horizon):
        true_value_indices = np.arange(num_eval_steps) + h
        predictions_at_h = predicted_forces[:num_eval_steps, h, :]
        true_values_at_h = actual_forces[true_value_indices, :]
        errors[:, h, :] = np.abs(predictions_at_h - true_values_at_h)

    print("Prediction errors calculated and statistics computed.")

    # --- 3. Calculate Statistics for Plotting ---
    mean_errors = np.mean(errors, axis=0)
    std_errors = np.std(errors, axis=0)

    mean_cd, mean_cl = mean_errors[:, 0], mean_errors[:, 1]
    std_cd, std_cl = std_errors[:, 0], std_errors[:, 1]

    horizon_steps = np.arange(1, horizon + 1)

    # --- 4. Create the Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True, constrained_layout=True)

    # --- Subplot for Drag Coefficient (Cd) ---
    axs[0].plot(horizon_steps, mean_cd, color='royalblue', lw=1.5, label='Mean error')
    axs[0].fill_between(horizon_steps, np.maximum(0, mean_cd - std_cd), mean_cd + std_cd, color='royalblue', alpha=0.2, label=r'Mean $\pm 1\sigma$')
    axs[0].set_ylabel('$L_1$ error ($C_d$)')
    axs[0].grid(True, which='both', linestyle=':', linewidth=0.5)
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(horizon_steps[0], horizon_steps[-1])
    axs[0].set_ylim(bottom=0)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # --- Subplot for Lift Coefficient (Cl) ---
    axs[1].plot(horizon_steps, mean_cl, color='royalblue', lw=1.5)
    axs[1].fill_between(horizon_steps, np.maximum(0, mean_cl - std_cl), mean_cl + std_cl, color='royalblue', alpha=0.2)
    axs[1].set_ylabel('$L_1$ error ($C_l$)')
    axs[1].set_xlabel('Prediction horizon step')
    axs[1].grid(True, which='both', linestyle=':', linewidth=0.5)
    axs[1].set_xlim(horizon_steps[0], horizon_steps[-1])
    axs[1].set_ylim(bottom=0)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)


    axs[1].set_xticks(np.arange(1, horizon + 1, 2))

    # --- 5. Save the Figure ---
    output_dir = Path("../04_Results/paper_figures/")
    base_filename = f"mpc_prediction_error_{hdf5_path.stem}"
    save_path_png = output_dir / f"{base_filename}.png"
    save_path_pdf = output_dir / f"{base_filename}.pdf"
    save_path_eps = output_dir / f"{base_filename}.eps"

    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    plt.savefig(save_path_eps)
    plt.close(fig)

    print(f"\nPlots successfully saved to:")
    print(f"  - {save_path_png}")
    print(f"  - {save_path_pdf}")
    print(f"  - {save_path_eps}")


if __name__ == "__main__":
    MPClogfile = '../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5'
    create_error_distribution_plot(input_file=MPClogfile)