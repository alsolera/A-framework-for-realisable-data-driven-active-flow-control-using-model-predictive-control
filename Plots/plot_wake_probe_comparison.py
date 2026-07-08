import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import re
from pathlib import Path
import matplotlib as mpl
import seaborn as sns  # Import seaborn

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})


def parse_openfoam_probe(filepath, probe_index):
    """
    Parses a single probe's data from an OpenFOAM probes file.

    Args:
        filepath (str or Path): The path to the OpenFOAM probes file.
        probe_index (int): The index of the probe to extract (e.g., 0).

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]:
            A tuple containing (time, velocity) NumPy arrays.
            Returns (None, None) if the probe is not found.
    """
    times = []
    velocities = []
    in_target_probe_section = False

    filepath = Path(filepath)
    # If the provided path is a directory, assume the data file is 'U' inside it.
    if filepath.is_dir():
        filepath = filepath / 'U'

    # Regex to find the vector in parentheses, e.g., (1.23 -4.56 7.89e-01)
    vector_pattern = re.compile(r'\((.*?)\)')

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Use regex for more robust header matching (handles variable spacing)
                header_match = re.search(r'# Probe\s+(\d+)', line)
                if header_match:
                    current_probe_idx = int(header_match.group(1))
                    if current_probe_idx == probe_index:
                        in_target_probe_section = True
                    else:
                        in_target_probe_section = False
                elif in_target_probe_section and line and not line.startswith('#'):
                    parts = line.split()
                    time = float(parts[0])

                    # Find the vector part of the string
                    match = vector_pattern.search(line)
                    if match:
                        vector_str = match.group(1)
                        velocity = [float(v) for v in vector_str.split()]

                        times.append(time)
                        velocities.append(velocity)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"An error occurred while parsing {filepath}: {e}")
        return None, None

    if not times:
        print(f"Warning: No data found for probe {probe_index} in {filepath}")
        return None, None

    return np.array(times), np.array(velocities)


def plot_uy_kde_comparison(vel_uc, vel_c, output_path, probe_index=0):
    """
    Creates and saves a minimalist KDE comparison for the Uy velocity component.

    Args:
        vel_uc (np.ndarray): Velocity data for the uncontrolled case (shape: N, 3).
        vel_c (np.ndarray): Velocity data for the controlled case (shape: M, 3).
        output_path (str or Path): Path to save the output plot (without extension).
        probe_index (int): The index of the probe being plotted.
    """
    uy_uc = vel_uc[:, 1]
    uy_c = vel_c[:, 1]

    # Calculate statistics for the legend
    std_uc = np.std(uy_uc)
    std_c = np.std(uy_c)

    # Smaller, less boxy figure size
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    # Plot KDEs
    sns.kdeplot(uy_uc, ax=ax, color='k', fill=True, alpha=0.7,
                linewidth=0, bw_adjust=0.5,
                label=f'Uncontrolled ($\\sigma={std_uc:.3f}$)')
    sns.kdeplot(uy_c, ax=ax, color='tab:blue', fill=True, alpha=0.7,
                linewidth=0, bw_adjust=0.5,
                label=f'Controlled ($\\sigma={std_c:.3f}$)')

    ax.set_xlabel('$U_y/U_\infty$')

    # Place legend in the center
    ax.legend(loc='upper center')

    # --- Remove visual clutter (spines, y-axis) ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)  # Hides y-axis ticks, labels, and spine

    plt.tight_layout()

    # Save in multiple formats for publication
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix('.png'), dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.eps'))

    # Corrected print statement
    print(f"Saved KDE comparison plot to {output_path.parent / (output_path.name + '.png/.pdf/.eps')}")
    plt.close(fig)


if __name__ == "__main__":
    # --- USER: PLEASE UPDATE THESE PATHS ---
    # Path to the probe data file for the UNCONTROLLED (baseline) case
    uncontrolled_probe_file = Path(
        "../gymprecice-run/jet_2Dtruck_DNSv3_20250831_1241_noAction/env_0/fluid-openfoam/postProcessing/probesDict_Uwake/0/")

    # Path to the probe data file for the CONTROLLED case (e.g., from an MPC run)
    controlled_probe_file = Path(
        "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20250906_223800/env_0/fluid-openfoam/postProcessing/probesDict_Uwake/100/")

    # The index of the probe you want to plot
    probe_to_plot = 0

    # Directory to save the final figure
    output_dir = Path("../04_Results/paper_figures/")
    output_filename_base = f"wake_probe_{probe_to_plot}_uy_kde"

    # --- Main script logic ---
    print("Parsing uncontrolled case data...")
    time_uncontrolled, vel_uncontrolled = parse_openfoam_probe(uncontrolled_probe_file, probe_to_plot)

    print("Parsing controlled case data...")
    time_controlled, vel_controlled = parse_openfoam_probe(controlled_probe_file, probe_to_plot)

    if vel_uncontrolled is not None and vel_controlled is not None:
        print("Data loaded successfully. Creating plot...")
        plot_uy_kde_comparison(
            vel_uc=vel_uncontrolled,
            vel_c=vel_controlled,
            output_path=output_dir / output_filename_base,
            probe_index=probe_to_plot
        )
        print("Done.")
    else:
        print("Could not proceed with plotting due to data loading errors.")