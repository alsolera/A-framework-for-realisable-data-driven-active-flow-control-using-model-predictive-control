from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import imageio
import shutil


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


def visualize_latent_space_mds_3d_multiview_and_gif(filepath, model_strdate_identifier, results_base_dir='04_Results',
                                                    suffix='', n_samples_for_mds=5000, make_gif=True):
    # Load data from file
    try:
        latent_space, forces_scaled, forces_mean, forces_std = load_file(filepath, printer=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # --- Unscale forces to get physical values ---
    forces_unscaled = forces_scaled * forces_std + forces_mean
    print("Using unscaled (physical) forces for coloring.")

    # --- Subsample the data for MDS ---
    if latent_space.shape[0] > n_samples_for_mds:
        print(f"Subsampling {n_samples_for_mds} points from {latent_space.shape[0]} for MDS...")
        idx = np.random.choice(latent_space.shape[0], n_samples_for_mds, replace=False)
        latent_subset = latent_space[idx, :]
        forces_subset = forces_unscaled[idx, :]
    else:
        print("Using full dataset for MDS.")
        latent_subset = latent_space
        forces_subset = forces_unscaled

    # --- Apply MDS ---
    n_components = 3
    print(f"Applying MDS to map latent space to {n_components}D...")
    mds = MDS(n_components=n_components, random_state=42, n_jobs=-1, normalized_stress='auto')
    z_mds = mds.fit_transform(latent_subset)
    print(f"MDS fitting complete. Stress = {mds.stress_:.4f}")

    results_dir = Path(results_base_dir) / model_strdate_identifier
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Plotting Function for Multi-View Static Figure ---
    def create_multiview_plot(data_3d, colors, label, output_path):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
        fig.suptitle(f'3D MDS Visualization of Latent Space - Colored by {label}', fontsize=16)

        viewpoints = [
            {'elev': 20, 'azim': -65, 'title': 'Perspective View'},
            {'elev': 90, 'azim': -90, 'title': 'Top-Down View (Z-axis)'},
            {'elev': 0, 'azim': -90, 'title': 'Side View (Y-axis)'}
        ]

        # Determine shared color limits
        vmin, vmax = np.min(colors), np.max(colors)

        for ax, view in zip(axs, viewpoints):
            scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=colors, cmap='coolwarm',
                                 alpha=0.7, s=15, edgecolor='none', vmin=vmin, vmax=vmax)
            ax.set_xlabel('MDS 1')
            ax.set_ylabel('MDS 2')
            ax.set_zlabel('MDS 3')
            ax.view_init(elev=view['elev'], azim=view['azim'])
            ax.set_title(view['title'])

        # Add a single colorbar for the whole figure
        fig.colorbar(scatter, ax=axs.ravel().tolist(), shrink=0.8, label=label)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300)
        plt.savefig(output_path.with_suffix('.pdf'))
        plt.close(fig)
        print(f"Saved 3D MDS multi-view plot to {output_path}")

    # --- Plotting Function for Rotating GIF ---
    def create_rotating_gif(data_3d, colors, label, output_path):
        print(f"Generating rotating GIF for {label}...")
        temp_dir = results_dir / "temp_gif_frames"
        temp_dir.mkdir(exist_ok=True)
        filenames = []

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=colors, cmap='coolwarm',
                             alpha=0.7, s=20, edgecolor='none')
        ax.set_xlabel('MDS Comp. 1')
        ax.set_ylabel('MDS Comp. 2')
        ax.set_zlabel('MDS Comp. 3')
        fig.colorbar(scatter, label=label, shrink=0.8)
        plt.tight_layout()

        for i, angle in enumerate(np.linspace(0, 360, 90, endpoint=False)):
            ax.view_init(elev=20, azim=angle)
            filename = temp_dir / f"frame_{i:03d}.png"
            plt.savefig(filename, dpi=150)
            filenames.append(filename)

        plt.close(fig)

        with imageio.get_writer(output_path, mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        shutil.rmtree(temp_dir)  # Clean up temporary frames
        print(f"Saved rotating GIF to {output_path}")

    # --- Create and Save Plots ---
    output_multiview_cd = results_dir / f"{model_strdate_identifier}{suffix}_MDS3D_multiview_Cd.png"
    create_multiview_plot(z_mds, forces_subset[:, 0], '$C_d$', output_multiview_cd)

    output_multiview_cl = results_dir / f"{model_strdate_identifier}{suffix}_MDS3D_multiview_Cl.png"
    create_multiview_plot(z_mds, forces_subset[:, 1], '$C_l$', output_multiview_cl)

    if make_gif:
        try:
            output_gif_cd = results_dir / f"{model_strdate_identifier}{suffix}_MDS3D_rotating_Cd.gif"
            create_rotating_gif(z_mds, forces_subset[:, 0], '$C_d$', output_gif_cd)

            output_gif_cl = results_dir / f"{model_strdate_identifier}{suffix}_MDS3D_rotating_Cl.gif"
            create_rotating_gif(z_mds, forces_subset[:, 1], '$C_l$', output_gif_cl)
        except ImportError:
            print("\nWarning: Could not create GIF. Please install imageio: 'pip install imageio'")
        except Exception as e:
            print(f"\nAn error occurred during GIF creation: {e}")

    return output_multiview_cd, output_multiview_cl


if __name__ == "__main__":
    file = '../04_Results/jet_2Dtruck_20250307_FMsignal_50000/20250906_19_38_58_LSTM_dim8_lb32_l1_h256_dr0.0_lr0.001_bs512_wC_ntest0_latent_space.hdf5'
    model_id = file.split('/')[-1].split('_LSTM_')[0]
    results_dir = file.split(model_id)[0]

    visualize_latent_space_mds_3d_multiview_and_gif(file,
                                                    model_id,
                                                    results_dir,
                                                    n_samples_for_mds=5000)