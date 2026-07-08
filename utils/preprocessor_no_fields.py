import os
import h5py
import numpy as np
from pathlib import Path


def find_latest_postprocessing_file(directory: Path, base_filename: str) -> Path:
    """
    Finds the largest file in a directory that matches a base name pattern.
    This handles cases where OpenFOAM creates e.g., `p` (empty) and `p_0`.
    """
    files = list(directory.glob(f"{base_filename}*"))
    if not files:
        raise FileNotFoundError(f"No files matching '{base_filename}*' found in {directory}")

    # Find the file with the maximum size
    latest_file = max(files, key=lambda p: p.stat().st_size)
    return latest_file


def get_folders(source_dir):
    # Function to check if a string is a valid float
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Function to sort folders by their numeric names
    def sort_folders(folders):
        numeric_folders = [folder for folder in folders if is_float(folder)]
        return sorted(numeric_folders, key=lambda x: float(x))

    return sort_folders(os.listdir(source_dir))


def load_pressure_file(filename, times):
    times_array = np.array(times, dtype=float)

    print(f'{len(times_array)} timesteps')

    # Load the data skipping the comment/header lines
    data = np.loadtxt(filename, comments='#')

    # Use searchsorted to get the closest indices
    indices = np.searchsorted(data[:, 0], times_array)

    # Handle cases where searchsorted might pick the next index instead of the closest
    indices = np.clip(indices, 1, len(data) - 1)
    left_indices = indices - 1
    right_indices = indices

    # Compare which of the two possible indices is closer
    left_distances = np.abs(data[left_indices, 0] - times_array)
    right_distances = np.abs(data[right_indices, 0] - times_array)

    indices = np.where(left_distances <= right_distances, left_indices, right_indices)
    # Filter data based on indices
    filtered_data = data[indices]

    deltas = filtered_data[:, 0] - times_array
    if max(deltas) > 0.01:
        print('WARNING: difference in time of p and U values')

    # print(f'Filtered p data: {filtered_data.shape}')
    print(f'Max time difference between p and U: {max(deltas):.4f}, mean: {np.mean(deltas):.6f}')

    return filtered_data[:, 1:]


def load_jet_file(filename, times):
    times_array = np.array(times, dtype=float)

    print(f'{len(times_array)} timesteps')

    # Load the data skipping the comment/header lines
    data = np.loadtxt(filename, comments='#')

    # Use searchsorted to get the closest indices
    indices = np.searchsorted(data[:, 0], times_array)

    # Handle cases where searchsorted might pick the next index instead of the closest
    indices = np.clip(indices, 1, len(data) - 1)
    left_indices = indices - 1
    right_indices = indices

    # Compare which of the two possible indices is closer
    left_distances = np.abs(data[left_indices, 0] - times_array)
    right_distances = np.abs(data[right_indices, 0] - times_array)

    indices = np.where(left_distances <= right_distances, left_indices, right_indices)

    # Filter data based on indices
    filtered_data = data[indices]

    deltas = filtered_data[:, 0] - times_array
    if max(deltas) > 0.01:
        print('WARNING: difference in time of jet and U values')

    print(f'Max time difference between jet and U: {max(deltas):.4f}, mean: {np.mean(deltas):.6f}')

    return filtered_data[:, 1:]


def load_forces_file(filename, times):
    # Function to extract header information from the file
    def extract_header_info(file_path):
        header_info = None
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('# Time'):
                    header_info = line.strip().split()[2:]
                    break
        return header_info

    times_array = np.array(times, dtype=float)

    print(f'{len(times_array)} timesteps')

    # Get column names from header
    header = extract_header_info(filename)

    # Load the data skipping the comment/header lines
    data = np.loadtxt(filename, comments='#')

    # Find indices of the closest timestamps
    # Use searchsorted to get the closest indices
    indices = np.searchsorted(data[:, 0], times_array)

    # Handle cases where searchsorted might pick the next index instead of the closest
    indices = np.clip(indices, 1, len(data) - 1)
    left_indices = indices - 1
    right_indices = indices

    # Compare which of the two possible indices is closer
    left_distances = np.abs(data[left_indices, 0] - times_array)
    right_distances = np.abs(data[right_indices, 0] - times_array)

    indices = np.where(left_distances <= right_distances, left_indices, right_indices)
    # Filter data based on indices
    filtered_data = data[indices]

    deltas = filtered_data[:, 0] - times_array
    if max(deltas) > 0.01:
        print('WARNING: difference in time of forces and U values')

    # print(f'Filtered forces data: {filtered_data.shape}')
    print(f'Max time difference between forces and U: {max(deltas):.4f}, mean: {np.mean(deltas):.6f}')

    return filtered_data[:, 1:], header


def parse_probe_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_idx = 0
    end_idx = 0
    for idx, line in enumerate(lines):
        if 'probeLocations' in line:
            start_idx = idx + 2
        if start_idx > 0 and ');' in line:
            end_idx = idx
            break

    coordinates = []
    for line in lines[start_idx:end_idx]:
        line = line.split('//')[0].strip()
        if line.startswith('(') and line.endswith(')'):
            coord_str = line[1:-1].split()
            coordinates.append([float(num) for num in coord_str])

    return np.array(coordinates)


def save_file(fname, Psensors, Pmean, Pstd, forces, forces_header, jet_actualFlow, jet_control,
              probe_coords):
    with h5py.File(fname, 'w') as f:
        f.create_dataset('Psensors', data=Psensors, dtype='float32')
        f.create_dataset('Pmean', data=Pmean, dtype='float32')
        f.create_dataset('Pstd', data=Pstd, dtype='float32')
        f.create_dataset('forces', data=forces, dtype='float32')
        f.create_dataset('forces_header', data=forces_header)
        f.create_dataset('jet_actualFlow', data=jet_actualFlow, dtype='float32')
        f.create_dataset('jet_control', data=jet_control, dtype='float32')
        f.create_dataset('probe_coords', data=probe_coords, dtype='float32')


def save_summary_csv(fname, timestamps, jet_control, jet_actualFlow, forces, forces_header):
    """
    Saves a human-readable CSV file with key time-series data.
    """
    # Find the indices for Cd and Cl from the forces_header
    try:
        # The header from load_forces_file is a list of strings like ['(sum', 'of', 'Cd)', '(sum', ...]
        # We need to find the index corresponding to the start of the 'Cd)' and 'Cl)' names.
        # A simple way is to find the strings and get their index. This is fragile but works for the known format.
        cd_col_index = [i for i, s in enumerate(forces_header) if 'Cd' in s][0]
        cl_col_index = [i for i, s in enumerate(forces_header) if 'Cl' in s][0]
    except IndexError:
        print("Warning: Could not find 'Cd' or 'Cl' in forces header. Using default columns 1 and 3.")
        # Defaulting to columns 1 (Cd) and 3 (Cl) if header parsing fails
        cd_col_index = 1
        cl_col_index = 3

    # Extract Cd and Cl columns
    cd_column = forces[:, cd_col_index]
    cl_column = forces[:, cl_col_index]

    # Stack the data columns for saving
    # Ensure all components are 1D arrays
    data_to_save = np.stack((
        timestamps,
        np.squeeze(jet_control),
        np.squeeze(jet_actualFlow),
        cd_column,
        cl_column
    ), axis=1)

    # Define header for the CSV file
    csv_header = "Time,Control,ActualFlow,Cd,Cl"

    # Save to CSV
    np.savetxt(fname, data_to_save, delimiter=',', header=csv_header, fmt='%.8f', comments='')
    print(f"Human-readable summary saved to: {fname}")


if __name__ == "__main__":
    # Source directory
    directory = '../gymprecice-run/jet_2Dtruck_DNSv3_20260329_1205_oodChirpU15_f001to08/env_0/fluid-openfoam/'
    name = 'jet_2Dtruck_20260329_1205_oodChirpU15_f001to08'
    dt = 0.2
    print(name)

    file_probes = directory.split('env_0/fluid-openfoam/')[0] + 'fluid-openfoam/system/probes'
    destFolder = directory.split('env_0/fluid-openfoam/')[0]

    # Get probe coordinates from file
    print('################# Loading probes coordinates')
    probe_coords = parse_probe_coordinates(file_probes)
    print(f'probe_coords shape: {probe_coords.shape}')

    # --- Determine valid timestamps from an OpenFOAM output file ---
    print('################# Determining valid timestamps')
    # Use flowRateJet1 as the reference for available time directories and file content
    jet_folder_ref = directory + 'postProcessing/flowRateJet1/'
    time_folders = get_folders(jet_folder_ref)
    if not time_folders:
        raise FileNotFoundError(f"No time folders found in {jet_folder_ref}")

    # We assume all postProcessing was written to the same time directories, so we use the first one found.
    # OpenFOAM might create multiple (e.g., 0, 300) but they should contain the full history up to that point.
    first_time_folder = time_folders[0]
    post_processing_dir = Path(jet_folder_ref) / first_time_folder

    # Dynamically find the correct data files
    jet_actual_flow_file = find_latest_postprocessing_file(post_processing_dir, "surfaceFieldValue")
    reference_time_file = jet_actual_flow_file # Use this for getting timestamps

    print(f"Reading all available timestamps from reference file: {reference_time_file}")

    # Get all timestamps present in the reference file
    all_available_times = np.loadtxt(reference_time_file, comments='#')[:, 0]

    # We want to sample the data at a consistent interval `dt`, matching the MPC control interval.
    start_time = 0.0  # Or set to a later time to skip initial transients, e.g., start_time = 2.0
    end_time = all_available_times[-1]

    timestamps_array_full = np.arange(start_time, end_time + dt / 2, dt)
    timestamps_array_full = timestamps_array_full[timestamps_array_full <= end_time]

    print(
        f"Generated {len(timestamps_array_full)} desired timestamps from t={start_time:.2f} to t={end_time:.2f} with dt={dt}")

    # --- Load full-length data first ---
    print('################# Loading full-length time series data')
    jet_actualFlow_full = load_jet_file(jet_actual_flow_file, timestamps_array_full)

    jet_control_dir = Path(directory) / 'postProcessing/flowRateJet2' / first_time_folder
    jet_control_file = find_latest_postprocessing_file(jet_control_dir, "surfaceFieldValue")
    jet_control_full = load_jet_file(jet_control_file, timestamps_array_full)

    p_sensors_dir = Path(directory) / 'postProcessing/probes' / first_time_folder
    p_sensors_file = find_latest_postprocessing_file(p_sensors_dir, "p")
    p_sensors_full = load_pressure_file(p_sensors_file, timestamps_array_full)

    forces_dir = Path(directory) / 'postProcessing/forceCoeffs' / first_time_folder
    forces_file = find_latest_postprocessing_file(forces_dir, "coefficient")
    forces_full, forces_header = load_forces_file(forces_file, timestamps_array_full)

    # --- Align data by shifting ---
    print('################# Aligning data by shifting time series')

    # Control actions from t_0 to t_{N-1}
    jet_control = jet_control_full[1:]

    # Resulting states from t_1 to t_N
    jet_actualFlow = jet_actualFlow_full[:-1]
    p_sensors = p_sensors_full[:-1]
    forces = forces_full[:-1]

    # Timestamps corresponding to the states
    timestamps_array = timestamps_array_full[:-1]

    nt = len(timestamps_array)
    print(f"Data aligned. Final number of samples: {nt}")

    print(f'jet_control shape: {jet_control.shape}')
    print(f'Sensors shape: {p_sensors.shape}')
    print(f'Forces: {forces.shape}, {forces_header}')

    p_mean = np.mean(p_sensors, axis=0)
    p_std = np.std(p_sensors)

    dest_filename_no_fields = f'{destFolder}{name}_{nt}_no_fields.hdf5'
    print(f"Saving to: {dest_filename_no_fields}")

    save_file(dest_filename_no_fields, p_sensors, p_mean, p_std,
              forces, forces_header, jet_actualFlow, jet_control, probe_coords)

    dest_filename_csv = f'{destFolder}{name}_{nt}_summary.csv'
    save_summary_csv(dest_filename_csv, timestamps_array, jet_control, jet_actualFlow, forces, forces_header)
