from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def load_file(path, printer=False):
    with h5py.File(path, 'r') as f:
        if 'latent_space' not in f:
            raise KeyError("HDF5 file must contain 'latent_space' dataset.")
        latent_space = f['latent_space'][:]

        if 'c_seq_all' in f:
            c_seq_all = f['c_seq_all'][:]
            min_len = min(latent_space.shape[0], c_seq_all.shape[0])
            latent_space = latent_space[:min_len]
            c_seq_all = c_seq_all[:min_len]
            forces_scaled = c_seq_all[:, 0, :]
        else:
            raise KeyError("Could not find 'c_seq_all' in the HDF5 file.")

        forces_mean = f['forces_mean'][()] if len(f['forces_mean'].shape) == 0 else f['forces_mean'][:]
        forces_std = f['forces_std'][()] if len(f['forces_std'].shape) == 0 else f['forces_std'][:]

    if printer:
        print('latent_space: ', latent_space.shape, latent_space.dtype)
        print('forces_scaled: ', forces_scaled.shape, forces_scaled.dtype)

    return latent_space, forces_scaled, forces_mean, forces_std


def visualize_latent_pairs(filepath, model_strdate_identifier, results_base_dir='04_Results', suffix='',
                           n_samples=5000, dataset_type='train'):
    # --- Select the correct file path based on the desired dataset type ---
    filepath_obj = Path(filepath)
    if dataset_type == 'eval':
        # Construct the expected name for the evaluation latent space file
        eval_filename = filepath_obj.name.replace('_latent_space.hdf5', '_eval_latent_space.hdf5')
        filepath_to_load = filepath_obj.with_name(eval_filename)
        if not filepath_to_load.exists():
            print(f"Error: Evaluation data file not found at expected path: {filepath_to_load}")
            return
        print(f"Loading EVALUATION data from: {filepath_to_load}")
    else:
        filepath_to_load = filepath_obj
        print(f"Loading TRAINING data from: {filepath_to_load}")

    # Load and prepare data
    try:
        latent_space, forces_scaled, forces_mean, forces_std = load_file(filepath_to_load, printer=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    forces_unscaled = forces_scaled * forces_std + forces_mean
    print("Using unscaled (physical) forces for coloring.")

    # Subsample the data for plotting clarity and performance
    if latent_space.shape[0] > n_samples:
        print(f"Subsampling {n_samples} points from {latent_space.shape[0]}...")
        idx = np.random.choice(latent_space.shape[0], n_samples, replace=False)
        latent_subset = latent_space[idx, :]
        forces_subset = forces_unscaled[idx, :]
    else:
        latent_subset = latent_space
        forces_subset = forces_unscaled

    num_dims = latent_subset.shape[1]

    # Determine consistent color limits
    vmin_cd, vmax_cd = np.min(forces_subset[:, 0]), np.max(forces_subset[:, 0])
    # Center Cl colormap around 0
    vmax_abs_cl = np.abs(forces_subset[:, 1]).max()
    vmin_cl, vmax_cl = -vmax_abs_cl, vmax_abs_cl

    # Create the figure with a more compact aspect ratio
    fig, axs = plt.subplots(num_dims, num_dims, figsize=(num_dims * 1., num_dims * 1.))

    scatter_cd = None
    scatter_cl = None

    # Determine axis limits for each mode to enforce consistency
    lims = []
    for d in range(num_dims):
        min_val, max_val = latent_subset[:, d].min(), latent_subset[:, d].max()
        padding = (max_val - min_val) * 0.05
        lims.append((min_val - padding, max_val + padding))

    for i in range(num_dims):
        for j in range(num_dims):
            ax = axs[i, j]

            # Diagonal: Plot histogram of the latent mode
            if i == j:
                ax.hist(latent_subset[:, i], bins='auto', color='gray', alpha=0.8)
                ax.set_yticklabels([])  # Hide tick labels on all histograms
                ax.tick_params(axis='y', length=0)  # Hide ticks on all histograms

            # Below diagonal: Color by Cd
            elif i > j:
                scatter_cd = ax.scatter(latent_subset[:, j], latent_subset[:, i],
                                        c=forces_subset[:, 0], cmap='viridis_r',
                                        s=3, alpha=0.7, vmin=vmin_cd, vmax=vmax_cd, rasterized=True)

            # Above diagonal: Color by Cl
            elif i < j:
                scatter_cl = ax.scatter(latent_subset[:, j], latent_subset[:, i],
                                        c=forces_subset[:, 1], cmap='coolwarm',
                                        s=3, alpha=0.7, vmin=vmin_cl, vmax=vmax_cl, rasterized=True)

            # Set consistent axis limits
            ax.set_xlim(lims[j])
            if i != j:
                ax.set_ylim(lims[i])

            # Set consistent axis limits manually
            ax.set_xlim(lims[j])
            if i != j: # Do not set y-limits for histograms
                ax.set_ylim(lims[i])

            # --- Label and Tick Management ---
            # Set labels ONLY on the outer edges
            if j == 0:
                ax.set_ylabel(f'Latent {i+1}')
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            if i == num_dims - 1:
                ax.set_xlabel(f'Latent {j+1}')
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

            # Hide interior tick labels and tick marks
            if i < num_dims - 1:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=0)
            if j > 0:
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)

    # Use subplots_adjust for fine-grained control over spacing and margins to make it compact
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.07, top=0.95, wspace=0.1, hspace=0.1)

    # Position colorbars in the right margin
    cbar_ax_cd = fig.add_axes([0.91, 0.1, 0.02, 0.35])
    if scatter_cd:
        cbar_cd = fig.colorbar(scatter_cd, cax=cbar_ax_cd)
        cbar_cd.set_label('$C_d$')

    cbar_ax_cl = fig.add_axes([0.91, 0.55, 0.02, 0.35])
    if scatter_cl:
        cbar_cl = fig.colorbar(scatter_cl, cax=cbar_ax_cl)
        cbar_cl.set_label('$C_l$')

    # Save the figure
    results_dir = Path(results_base_dir) / model_strdate_identifier
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{model_strdate_identifier}{suffix}_latent_pair_plot.png"

    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved pair plot to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    # --- User Configuration ---
    # Set this to 'train' or 'eval' to choose which dataset to plot
    DATASET_TO_PLOT = 'train'

    file = '../04_Results/jet_2Dtruck_20250307_FMsignal_50000/20250906_19_38_58_LSTM_dim8_lb32_l1_h256_dr0.0_lr0.001_bs512_wC_ntest0_latent_space.hdf5'
    model_id = file.split('/')[-1].split('_LSTM_')[0]
    results_dir = file.split(model_id)[0]

    # Generate a suffix for the output filename if plotting eval data
    plot_suffix = f'_{DATASET_TO_PLOT}' if DATASET_TO_PLOT == 'eval' else ''

    visualize_latent_pairs(file,
                           model_id,
                           results_dir,
                           n_samples=5000,
                           dataset_type=DATASET_TO_PLOT,
                           suffix=plot_suffix)
