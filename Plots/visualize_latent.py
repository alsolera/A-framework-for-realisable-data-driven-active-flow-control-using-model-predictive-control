from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA


def load_file(path, printer=False):
    with h5py.File(path, 'r') as f:
        if 'latent_space' not in f:
            raise KeyError("HDF5 file must contain 'latent_space' dataset.")
        latent_space = f['latent_space'][:]

        # The new save format stores sequences. We need 'c_seq_all'.
        if 'c_seq_all' in f:
            # Shape is (num_samples, lookforward_steps, num_force_coeffs)
            c_seq_all = f['c_seq_all'][:]
            if c_seq_all.shape[0] != latent_space.shape[0]:
                print(
                    f"Warning: Mismatch between latent space samples ({latent_space.shape[0]}) and force sequences ({c_seq_all.shape[0]}). Truncating to match.")
                min_len = min(latent_space.shape[0], c_seq_all.shape[0])
                latent_space = latent_space[:min_len]
                c_seq_all = c_seq_all[:min_len]

            # For visualization, use the force at the first prediction step (t+1)
            # This results in shape (num_samples, num_force_coeffs)
            forces_scaled = c_seq_all[:, 0, :]
        else:
            # Fallback for old file formats or if c_seq_all was not saved
            raise KeyError("Could not find 'c_seq_all' in the HDF5 file. The file format may be outdated or incorrect.")

        # Load scalar or array data correctly for mean and std
        forces_mean = f['forces_mean'][()] if len(f['forces_mean'].shape) == 0 else f['forces_mean'][:]
        forces_std = f['forces_std'][()] if len(f['forces_std'].shape) == 0 else f['forces_std'][:]

    if printer:
        print('latent_space: ', latent_space.shape, latent_space.dtype)
        print('forces_scaled: ', forces_scaled.shape, forces_scaled.dtype)
        # Handle printing for scalar or array
        print('forces_mean: ', forces_mean.shape if hasattr(forces_mean, 'shape') else 'scalar', forces_mean)
        print('forces_std: ', forces_std.shape if hasattr(forces_std, 'shape') else 'scalar', forces_std)

    return latent_space, forces_scaled, forces_mean, forces_std


def visualize_latent_space(filepath, model_strdate_identifier, results_base_dir='04_Results', suffix='',
                           axis_limits=None, c_limits_cd=None, c_limits_cl=None, pca_object=None):
    # Load data from file
    try:
        latent_space, forces_scaled, forces_mean, forces_std = load_file(filepath, printer=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Please ensure the file path is correct and the file exists.")
        exit()

    # Unscale forces to use for the color map, showing the true physical values
    forces_unscaled = forces_scaled * forces_std + forces_mean

    # Apply PCA: fit a new one for training data, or use existing one for eval data
    if pca_object is None:
        print("Fitting a new PCA transformation.")
        pca = PCA(n_components=3)
        z_pca = pca.fit_transform(latent_space)
    else:
        print("Using the provided PCA transformation.")
        pca = pca_object
        z_pca = pca.transform(latent_space)

    # Determine axis and color limits
    if axis_limits is None:
        x_lim = (z_pca[:, 0].min(), z_pca[:, 0].max())
        y_lim = (z_pca[:, 1].min(), z_pca[:, 1].max())
        z_lim = (z_pca[:, 2].min(), z_pca[:, 2].max())
        axis_limits = {'x': x_lim, 'y': y_lim, 'z': z_lim}

    # Use provided color limits or calculate them from the UNscaled data
    vmin_cd, vmax_cd = c_limits_cd if c_limits_cd is not None else (forces_unscaled[:, 0].min(),
                                                                    forces_unscaled[:, 0].max())
    vmin_cl, vmax_cl = c_limits_cl if c_limits_cl is not None else (forces_unscaled[:, 1].min(),
                                                                    forces_unscaled[:, 1].max())
    # Store calculated limits to return them
    c_limits_cd_out = (vmin_cd, vmax_cd)
    c_limits_cl_out = (vmin_cl, vmax_cl)

    # --- Create a single figure with two subplots for side-by-side comparison ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': '3d'})

    # --- Plot 1: Colored by Cd ---
    scatter1 = ax1.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], c=forces_unscaled[:, 0], cmap='coolwarm', alpha=0.7,
                           vmin=vmin_cd, vmax=vmax_cd, s=8)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')
    # ax1.set_title('Color = $C_d$') # Removed for cleaner plot
    ax1.set_xlim(axis_limits['x'])
    ax1.set_ylim(axis_limits['y'])
    ax1.set_zlim(axis_limits['z'])
    # Use MaxNLocator to reduce the number of ticks for a cleaner look
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax1.zaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.colorbar(scatter1, ax=ax1, shrink=0.6, label='$C_d$')

    # --- Plot 2: Colored by Cl ---
    scatter2 = ax2.scatter(z_pca[:, 0], z_pca[:, 1], z_pca[:, 2], c=forces_unscaled[:, 1], cmap='coolwarm', alpha=0.7,
                           vmin=vmin_cl, vmax=vmax_cl, s=8)
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    # ax2.set_title('Color = $C_l$') # Removed for cleaner plot
    ax2.set_xlim(axis_limits['x'])
    ax2.set_ylim(axis_limits['y'])
    ax2.set_zlim(axis_limits['z'])
    # Use MaxNLocator to reduce the number of ticks for a cleaner look
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.zaxis.set_major_locator(MaxNLocator(nbins=3))
    fig.colorbar(scatter2, ax=ax2, shrink=0.6, label='$C_l$')

    # --- Set a consistent viewing angle for both plots ---
    view_elevation = 20
    view_azimuth = -120
    ax1.view_init(elev=view_elevation, azim=view_azimuth)
    ax2.view_init(elev=view_elevation, azim=view_azimuth)

    # Create the results directory and save the combined plot
    results_dir = Path(results_base_dir) / model_strdate_identifier
    results_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = results_dir / f"{model_strdate_identifier}{suffix}_PCA_Cd_Cl.png"

    # Adjust subplot parameters to add more space between the plots
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0, top=1, wspace=0.1)
    plt.savefig(output_plot_path, format='png', dpi=300)
    plt.savefig(output_plot_path.with_suffix('.pdf'), format='pdf')
    plt.savefig(output_plot_path.with_suffix('.eps'), format='eps')
    plt.close(fig)
    print(f"Saved combined PCA plot to {output_plot_path}")

    return output_plot_path, axis_limits, c_limits_cd_out, c_limits_cl_out, pca


if __name__ == "__main__":
    file = '../04_Results/jet_2Dtruck_20250307_FMsignal_50000/20250906_13_49_13_LSTM_dim4_lb32_l1_h128_dr0.0_lr0.001_bs4096_wC_ntest0_latent_space.hdf5'
    model_id = file.split('/')[-1].split('_LSTM')[0]
    results_dir = file.split(model_id)[0]

    visualize_latent_space(file,
                           model_id,
                           results_dir)