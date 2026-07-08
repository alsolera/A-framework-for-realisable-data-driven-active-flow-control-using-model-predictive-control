import h5py
import numpy as np
import torch
import json


def prepare_full_dataset(args):
    # Load concatenated data and file boundaries
    (p_unscaled, forces_unscaled, control_unscaled, p_mean, p_std, forces_mean, forces_std,
     control_mean, control_std, probes, file_end_indices) = loadData(args)

    # Scale all concatenated data using scaling params from the first file
    p_scaled = (p_unscaled - p_mean) / p_std
    forces_scaled = (forces_unscaled - forces_mean) / forces_std
    control_scaled = (control_unscaled - control_mean) / control_std

    # --- Create sequences, avoiding crossing file boundaries ---
    all_s, all_a, all_s1, all_C = [], [], [], []

    start_offset = 0
    # Process each file's data segment separately to create sequences
    for end_idx in file_end_indices:
        p_segment = p_scaled[start_offset:end_idx]
        control_segment = control_scaled[start_offset:end_idx]
        forces_segment = forces_scaled[start_offset:end_idx]

        # Create sequences only within this contiguous data block
        s_t, a_t, s_t1, C_t = create_RNN_dataset(p_segment, control_segment, forces_segment, args.lookback,
                                                 args.recursive_train_steps)

        if s_t.shape[0] > 0:  # Only append if sequences were created
            all_s.append(s_t)
            all_a.append(a_t)
            all_s1.append(s_t1)
            all_C.append(C_t)

        start_offset = end_idx  # Move to the start of the next segment

    # Concatenate all generated sequences
    s_all = np.concatenate(all_s, axis=0)
    a_all = np.concatenate(all_a, axis=0)
    s_t1_all = np.concatenate(all_s1, axis=0)
    c_all = np.concatenate(all_C, axis=0)

    print(f'p: {p_scaled.shape}, control: {control_scaled.shape}, coeff: {forces_scaled.shape}')
    return s_all, a_all, s_t1_all, c_all


def create_RNN_dataset(sensors, control, forces, lookback, lookforward=1):
    """
    Prepare a time series dataset for training PLDM by transforming it into
    (s_t, a_seq, s_t1, c_seq) format.

    s_t is the sequence of sensor readings of length `lookback`.
    a_seq is the sequence of future control inputs of length `lookforward`.
    s_t1 is the sequence of sensor readings for the next lookforward time steps.
    c_seq are the force coefficients for the next lookforward time steps.
    """
    s_t, a_seq, s_t1, c_seq = [], [], [], []

    # The loop range needs to accommodate the lookforward for forces as well
    # Max starting index i is such that i+lookforward-1 is the last valid index
    for i in range(lookback, sensors.shape[0] - lookforward + 1):
        sensors_seq = sensors[i - lookback:i]           # (lookback, num_sensors)
        future_sensors_seq = sensors[i:i+lookforward]      # (lookforward, num_sensors)
        future_controls_seq = control[i:i+lookforward]     # (lookforward, control_dim)
        future_forces_seq = forces[i:i+lookforward]        # Get sequence of future forces (lookforward, num_forces)

        s_t.append(sensors_seq)
        a_seq.append(future_controls_seq)
        s_t1.append(future_sensors_seq)
        c_seq.append(future_forces_seq)

    return np.array(s_t), np.array(a_seq), np.array(s_t1), np.array(c_seq)


def _augment_with_symmetry(p_segment, forces_segment, control_segment, sel_coefs, sensor_map):
    """Augments a data segment by applying physical symmetry."""
    if sensor_map is None:
        return p_segment, forces_segment, control_segment

    # 1. Create flipped sensors
    # The sensor map is a list/array where sensor_map[i] is the partner of i
    p_flipped = p_segment[:, sensor_map]

    # 2. Create flipped forces (flip sign of Cl)
    forces_flipped = forces_segment.copy()
    try:
        # Find the index of the 'Cl' column in the selected coefficients
        cl_index = sel_coefs.index('Cl')
        forces_flipped[:, cl_index] *= -1
    except (ValueError, IndexError):
        print("Warning: Could not find 'Cl' in sel_coefs for symmetry augmentation. Skipping force flip.")

    # 3. Create flipped control (flip sign)
    control_flipped = control_segment.copy()
    control_flipped *= -1

    # 4. Concatenate original and flipped data
    p_augmented = np.vstack([p_segment, p_flipped])
    forces_augmented = np.vstack([forces_segment, forces_flipped])
    control_augmented = np.vstack([control_segment, control_flipped])

    return p_augmented, forces_augmented, control_augmented


def loadData(args, sel_coefs=['Cd', 'Cl'], printer=False):
    """
    Load data from one or more HDF5 files, concatenate them, and return raw data
    along with scaling parameters calculated from the FULL, final dataset.
    """
    files = args.datafile
    if isinstance(files, str):
        files = [files]  # Ensure it's always a list

    all_p_raw = []
    all_forces_raw = []
    all_control_raw = []
    file_end_indices = []
    total_samples = 0
    probes = None

    # --- Load Symmetry Map ---
    sensor_map = None
    if args.augment_with_symmetry:
        try:
            with open('sensor_symmetry_map.json', 'r') as f:
                map_dict = json.load(f)
                # Convert to a list for direct indexing, assuming keys are "0", "1", ...
                num_sensors = len(map_dict)
                sensor_map = [0] * num_sensors
                for k, v in map_dict.items():
                    sensor_map[int(k)] = v
        except FileNotFoundError:
            print("Warning: `augment_with_symmetry` is True, but `sensor_symmetry_map.json` not found. Augmentation will be skipped.")

    # --- Step 1: Load all raw data from files ---
    print(f"Loading and concatenating data from {len(files)} file(s)...")
    for i, file_path in enumerate(files):
        with h5py.File(file_path, 'r') as f:
            if i == 0: # Load probes and header only from the first file
                probes = f['probe_coords'][:]
                forces_header = [item.decode('utf-8') for item in f['forces_header'][:]]
                sel_coefs_idx = [forces_header.index(item) for item in sel_coefs]
                print(f'Selected coefficients indexes: {sel_coefs_idx}')

            p_file = f['Psensors'][:]
            forces_file = f['forces'][:, sel_coefs_idx]
            control_file = f['jet_control'][:]
            if control_file.ndim == 1:
                control_file = control_file[:, np.newaxis]

            # --- Apply Symmetry Augmentation to this file's data ---
            if args.augment_with_symmetry and sensor_map:
                p_file, forces_file, control_file = _augment_with_symmetry(
                    p_file, forces_file, control_file, sel_coefs, sensor_map
                )

            all_p_raw.append(p_file)
            all_forces_raw.append(forces_file)
            all_control_raw.append(control_file)

            num_samples_in_file = len(p_file)
            total_samples += num_samples_in_file
            file_end_indices.append(total_samples)

    # --- Step 2: Concatenate all raw data into final arrays ---
    p_unscaled = np.concatenate(all_p_raw, axis=0)
    forces_unscaled = np.concatenate(all_forces_raw, axis=0)
    control_unscaled = np.concatenate(all_control_raw, axis=0)

    # --- Step 3: Calculate scaling parameters from the COMPLETE dataset ---
    p_mean = np.mean(p_unscaled, axis=0)
    p_std = np.std(p_unscaled) # Global std for sensors
    forces_mean = np.mean(forces_unscaled, axis=0)
    forces_std = np.std(forces_unscaled, axis=0)
    control_mean = np.mean(control_unscaled, axis=0)
    control_std = np.std(control_unscaled, axis=0)

    if printer:
        print(f'Total final samples (after augmentation): {p_unscaled.shape[0]}')
        print(f'Calculated scaling params from full dataset - p_mean shape: {p_mean.shape}, p_std: {p_std}')
        print(f'Forces mean: {forces_mean}, Forces std: {forces_std}')


    return (p_unscaled, forces_unscaled, control_unscaled,
            p_mean, p_std, forces_mean, forces_std, control_mean, control_std,
            probes, file_end_indices)


def get_prepared_data(args, device, shuffle_train=True, shuffle_test=False):
    s_all, a_seq_all, s_t1_all, c_all = prepare_full_dataset(args)

    # --- Explicit Train/Test Split ---
    n_total = s_all.shape[0]
    if args.n_test == 0:
        n_train = n_total
        s_train, a_seq_train, s_t1_train, c_train = s_all, a_seq_all, s_t1_all, c_all
        s_test, a_seq_test, s_t1_test, c_test = (np.array([]), np.array([]), np.array([]), np.array([]))
    else:
        n_train = n_total - args.n_test
        s_train, a_seq_train, s_t1_train, c_train = s_all[:n_train], a_seq_all[:n_train], s_t1_all[:n_train], c_all[:n_train]
        s_test, a_seq_test, s_t1_test, c_test = s_all[n_train:], a_seq_all[n_train:], s_t1_all[n_train:], c_all[n_train:]

    print(f"N train: {s_train.shape[0]} + N test: {s_test.shape[0]} = N total: {n_total}")
    print(f'Train: s_t={s_train.shape}, a_t={a_seq_train.shape}, s_t1={s_t1_train.shape}, C_t={c_train.shape}')
    print(f'Test: s_t={s_test.shape}, a_t={a_seq_test.shape}, s_t1={s_t1_test.shape}, C_t={c_test.shape}')


    # Convert to PyTorch tensors
    s_train, s_test = torch.tensor(s_train, dtype=torch.float32), torch.tensor(s_test, dtype=torch.float32)
    a_seq_train, a_seq_test = torch.tensor(a_seq_train, dtype=torch.float32), torch.tensor(a_seq_test, dtype=torch.float32)
    s_t1_train, s_t1_test = torch.tensor(s_t1_train, dtype=torch.float32), torch.tensor(s_t1_test, dtype=torch.float32)
    c_train, c_test = torch.tensor(c_train, dtype=torch.float32), torch.tensor(c_test, dtype=torch.float32)

    # Move data to GPU if specified

    if args.DATA_TO_GPU and device.type == 'cuda':
        s_train, a_seq_train, s_t1_train, c_train = s_train.to(device), a_seq_train.to(device), s_t1_train.to(device), c_train.to(device)
        s_test, a_seq_test, s_t1_test, c_test = s_test.to(device), a_seq_test.to(device), s_t1_test.to(device), c_test.to(device)

    # PyTorch Dataset
    dataset_train = torch.utils.data.TensorDataset(s_train, a_seq_train, s_t1_train, c_train)
    dataset_test = torch.utils.data.TensorDataset(s_test, a_seq_test, s_t1_test, c_test)

    # Dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=shuffle_train
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=shuffle_test
    )

    return dataloader_train, dataloader_test



def prepare_raw_data(args):
    """
    Load and scale data from files but do NOT create sequences yet.
    Returns the scaled arrays and metadata needed to later build dataloaders.
    This is the expensive I/O step — call it once before the training loop.

    Returns
    -------
    p_scaled        : np.ndarray  (T, n_sensors)  — normalised sensor readings
    control_scaled  : np.ndarray  (T, control_dim)
    forces_scaled   : np.ndarray  (T, n_forces)
    file_end_indices: list[int]   — segment boundaries (for sequence creation)
    """
    (p_unscaled, forces_unscaled, control_unscaled,
     p_mean, p_std, forces_mean, forces_std,
     control_mean, control_std, probes, file_end_indices) = loadData(args)

    p_scaled       = (p_unscaled       - p_mean)       / p_std
    forces_scaled  = (forces_unscaled  - forces_mean)  / forces_std
    control_scaled = (control_unscaled - control_mean) / control_std

    print(f'prepare_raw_data: p={p_scaled.shape}, control={control_scaled.shape}, '
          f'forces={forces_scaled.shape}')
    return p_scaled, control_scaled, forces_scaled, file_end_indices


def build_dataloader_from_scaled(
        p_scaled, control_scaled, forces_scaled, file_end_indices,
        args, device, noise_sigma=0.0, rng=None,
        shuffle_train=True, shuffle_test=False):
    """
    Build train/test dataloaders from already-scaled sensor arrays.
    Optionally adds fresh Gaussian noise to p_scaled (noise applied to the
    full time series BEFORE sequence creation, so every sequence that contains
    timestep t sees the same noise realisation for t).

    Parameters
    ----------
    noise_sigma : float
        Std of additive Gaussian noise in normalised sensor units.
        Pass 0.0 for no noise.
    rng : np.random.Generator or None
        Random generator for reproducible noise. If None, uses np.random.
    """
    # --- Inject noise on the full time series (before slicing into sequences) ---
    if noise_sigma > 0.0:
        generator = rng if rng is not None else np.random
        noise = generator.normal(0.0, noise_sigma, size=p_scaled.shape).astype(np.float32)
        p_noisy = p_scaled + noise
    else:
        p_noisy = p_scaled

    # --- Create sequences, respecting file boundaries ---
    all_s, all_a, all_s1, all_C = [], [], [], []
    start_offset = 0
    for end_idx in file_end_indices:
        s_t, a_t, s_t1, C_t = create_RNN_dataset(
            p_noisy[start_offset:end_idx],
            control_scaled[start_offset:end_idx],
            forces_scaled[start_offset:end_idx],
            args.lookback, args.recursive_train_steps)
        if s_t.shape[0] > 0:
            all_s.append(s_t);  all_a.append(a_t)
            all_s1.append(s_t1); all_C.append(C_t)
        start_offset = end_idx

    s_all   = np.concatenate(all_s,  axis=0)
    a_all   = np.concatenate(all_a,  axis=0)
    s_t1_all= np.concatenate(all_s1, axis=0)
    c_all   = np.concatenate(all_C,  axis=0)

    # --- Train / test split ---
    n_total = s_all.shape[0]
    if args.n_test == 0:
        n_train = n_total
        s_train, a_train, s_t1_train, c_train = s_all, a_all, s_t1_all, c_all
        s_test = a_test = s_t1_test = c_test = np.array([])
    else:
        n_train = n_total - args.n_test
        s_train,   a_train,   s_t1_train,   c_train   = s_all[:n_train],  a_all[:n_train],  s_t1_all[:n_train],  c_all[:n_train]
        s_test,    a_test,    s_t1_test,    c_test    = s_all[n_train:],  a_all[n_train:],  s_t1_all[n_train:],  c_all[n_train:]

    # --- Convert to tensors ---
    s_train   = torch.tensor(s_train,   dtype=torch.float32)
    a_train   = torch.tensor(a_train,   dtype=torch.float32)
    s_t1_train= torch.tensor(s_t1_train,dtype=torch.float32)
    c_train   = torch.tensor(c_train,   dtype=torch.float32)

    if args.DATA_TO_GPU and device.type == 'cuda':
        s_train, a_train, s_t1_train, c_train = (
            s_train.to(device), a_train.to(device),
            s_t1_train.to(device), c_train.to(device))

    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(s_train, a_train, s_t1_train, c_train),
        batch_size=args.batch_size, shuffle=shuffle_train)

    # Test dataloader (only built if n_test > 0)
    dl_test = None
    if args.n_test > 0 and len(s_test) > 0:
        s_test_t    = torch.tensor(s_test,    dtype=torch.float32)
        a_test_t    = torch.tensor(a_test,    dtype=torch.float32)
        s_t1_test_t = torch.tensor(s_t1_test, dtype=torch.float32)
        c_test_t    = torch.tensor(c_test,    dtype=torch.float32)
        if args.DATA_TO_GPU and device.type == 'cuda':
            s_test_t, a_test_t, s_t1_test_t, c_test_t = (
                s_test_t.to(device), a_test_t.to(device),
                s_t1_test_t.to(device), c_test_t.to(device))
        dl_test = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(s_test_t, a_test_t, s_t1_test_t, c_test_t),
            batch_size=args.batch_size, shuffle=shuffle_test)

    return dl_train, dl_test


'''Testing code'''
if __name__ == "__main__":

    (p_scaled, p_mean, p_std, forces_scaled, forces_mean, forces_std, control_scaled, control_mean, control_std, probes) =(
        loadData('../01_Data/jet_2Dtruck_DNSv3_20241205_1931_filteredGaussianU15_f005to05_50000_noFields.hdf5', printer=True))