import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
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

def parse_openfoam_pressure_probes(filepath, probe_indices_to_average, start_time=0.0):
    """
    Parses an OpenFOAM pressure probes file, which contains multiple probes'
    scalar data in columns for each time step. It calculates the mean value
    across a specified subset of probes and filters by a start time.

    Args:
        filepath (str or Path): Path to the OpenFOAM pressure probes file (e.g., 'p').
        probe_indices_to_average (list[int]): A list of probe indices to select
                                               and average (e.g., [80, 81, ..., 89]).
        start_time (float): The time from which to start including data.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]:
            A tuple containing (time, mean_pressure) NumPy arrays.
            Returns (None, None) if parsing fails.
    """
    filepath = Path(filepath)
    # If the provided path is a directory, assume the data file is 'p' inside it.
    if filepath.is_dir():
        filepath = filepath / 'p'

    data_lines = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Skip any comment lines
                if line.strip().startswith('#'):
                    continue
                data_lines.append(line.strip().split())

        if not data_lines:
            print(f"Warning: No data lines found in {filepath}")
            return None, None

        # Convert to a NumPy array of floats
        raw_data = np.array(data_lines, dtype=float)

        # Filter data by start time first
        time_filtered_data = raw_data[raw_data[:, 0] >= start_time]

        if time_filtered_data.shape[0] == 0:
            print(f"Warning: No data found after start_time={start_time} in {filepath}")
            return None, None

        # First column is time
        times = time_filtered_data[:, 0]

        # Subsequent columns are p_probe_0, p_probe_1, ...
        # Column index for a probe is probe_index + 1
        column_indices_to_average = [i + 1 for i in probe_indices_to_average]

        # Select the data for the specified probes
        selected_probe_data = time_filtered_data[:, column_indices_to_average]

        # Calculate the mean across the selected probes for each time step
        mean_pressures = np.mean(selected_probe_data, axis=1)

        return times, mean_pressures

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except IndexError:
        print(f"Error: A probe index in {probe_indices_to_average} might be out of bounds for the data in {filepath}.")
        return None, None
    except Exception as e:
        print(f"An error occurred while parsing {filepath}: {e}")
        return None, None


def plot_pressure_kde_comparison(pressure_uc, pressure_c, output_path):
    """
    Creates and saves a minimalist KDE comparison for mean back pressure.

    Args:
        pressure_uc (np.ndarray): Mean pressure data for the uncontrolled case.
        pressure_c (np.ndarray): Mean pressure data for the controlled case.
        output_path (str or Path): Path to save the output plot (without extension).
    """
    # Calculate statistics for the legend
    mean_uc = np.mean(pressure_uc)
    mean_c = np.mean(pressure_c)

    # Smaller, less boxy figure size
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    # Plot KDEs using seaborn, with increased bandwidth and no contour line
    sns.kdeplot(pressure_uc, ax=ax, color='k', fill=True, alpha=0.7,
                linewidth=0,
                label=rf'Uncontrolled ($\bar{{C_p}}={mean_uc:.3f}$)')
    sns.kdeplot(pressure_c, ax=ax, color='tab:blue', fill=True, alpha=0.7,
                linewidth=0,
                label=rf'Controlled ($\bar{{C_p}}={mean_c:.3f}$)')

    ax.set_xlabel(f'Mean base $C_p$')

    # Place legend in the center
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.05))

    # --- Remove visual clutter (spines, y-axis) ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.set_ylabel('Density')

    plt.tight_layout()

    # Save in multiple formats for publication
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix('.png'), dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))

    print(f"Saved pressure KDE plot to {output_path.parent / (output_path.name + '.png/.pdf/.eps')}")
    plt.close(fig)


if __name__ == "__main__":
    # --- USER: PLEASE UPDATE THESE PATHS ---
    # NOTE: These paths should point to the directory containing the 'p' file, or the 'p' file itself.

    # Path for the UNCONTROLLED (baseline) case pressure probes
    uncontrolled_probe_file = Path(
        "../gymprecice-run/jet_2Dtruck_DNSv3_20250831_1241_noAction/env_0/fluid-openfoam/postProcessing/probes/0/")

    # Path for the CONTROLLED case pressure probes
    controlled_probe_file = Path(
        "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20250906_223800/env_0/fluid-openfoam/postProcessing/probes/0/")

    # Define the indices of the probes on the back of the truck
    back_probes_indices = list(range(80, 90))  # Probes 80 to 89

    # Directory to save the final figure
    output_dir = Path("../04_Results/paper_figures/")
    output_filename_base = "back_pressure_kde_comparison"  # Updated filename

    # --- USER: VERIFY THESE PHYSICAL CONSTANTS for Cp CALCULATION ---
    # These should match the values from your simulation setup.
    rho = 1.0      # Fluid density [kg/m^3]
    U_inf = 1.0    # Freestream velocity [m/s]
    p_inf = 0.0      # Freestream static pressure [Pa]

    # Calculate the dynamic pressure (q_inf = 0.5 * rho * U^2) for non-dimensionalization
    dyn_pressure = 0.5 * rho * U_inf**2

    # --- Main script logic ---
    print(f"Parsing uncontrolled case pressure data for probes {back_probes_indices}...")
    time_uncontrolled, pressure_uncontrolled = parse_openfoam_pressure_probes(
        uncontrolled_probe_file,
        back_probes_indices
    )

    print(f"Parsing controlled case pressure data for probes {back_probes_indices} (from t=100s onwards)...")
    time_controlled, pressure_controlled = parse_openfoam_pressure_probes(
        controlled_probe_file,
        back_probes_indices,
        start_time=100.0  # Apply time filter for the controlled case
    )

    # Proceed only if both datasets were loaded successfully
    if pressure_uncontrolled is not None and pressure_controlled is not None and dyn_pressure > 1e-9:
        # Convert raw pressure values to Pressure Coefficient (Cp)
        cp_uncontrolled = (pressure_uncontrolled - p_inf) / dyn_pressure
        cp_controlled = (pressure_controlled - p_inf) / dyn_pressure

        print("Data loaded successfully. Creating plot...")
        plot_pressure_kde_comparison(
            pressure_uc=cp_uncontrolled,
            pressure_c=cp_controlled,
            output_path=output_dir / output_filename_base
        )
        print("Done.")
    else:
        print("Could not proceed with plotting due to data loading errors.")