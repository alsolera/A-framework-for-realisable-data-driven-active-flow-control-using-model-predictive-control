# inspect_h5.py

import h5py
import argparse
import numpy as np
from pathlib import Path


def inspect_h5_file(filepath: Path):
    """
    Opens an HDF5 file and prints the name, shape, dtype, and summary statistics
    for each numerical dataset.
    """
    if not filepath.is_file():
        print(f"Error: File not found at '{filepath}'")
        return

    print(f"\n--- Inspecting file: {filepath.name} ---")

    try:
        with h5py.File(filepath, 'r') as f:
            if not f.keys():
                print("File is empty or contains no datasets at the root level.")
                return

            print("Datasets found:")

            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Basic Info: Name, Shape, Dtype
                    print(f"  - {name:<20} | Shape: {str(obj.shape):<20} | Dtype: {obj.dtype}")

                    # Check if the dataset is likely numerical and not empty
                    if np.issubdtype(obj.dtype, np.number) and obj.size > 0:
                        # Load data into memory to compute stats. For very large datasets,
                        # you might want to sample, but for this use case it's fine.
                        data = obj[()]

                        # Use try-except to handle potential non-numeric data gracefully if check fails
                        try:
                            mean_val = np.mean(data)
                            std_val = np.std(data)
                            min_val = np.min(data)
                            max_val = np.max(data)

                            # Print stats on a new line for clarity
                            print(
                                f"    {'Stats:':<20}   Mean={mean_val:<12.4f} Std={std_val:<12.4f} Min={min_val:<12.4f} Max={max_val:<12.4f}")
                        except (TypeError, ValueError):
                            # This might happen if a dataset has a numeric dtype but contains non-numeric values
                            print(f"    {'Stats:':<20}   Could not compute statistics (possibly non-numeric data).")

            f.visititems(print_dataset_info)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


if __name__ == "__main__":

    basedir = '../'
    filepaths = ['01_Data/jet_2Dtruck_DNSv3_20250307_2155_FMsignal_01_50000_unscaledSensors_50000_no_fields.hdf5',
                 'gymprecice-run/jet_2Dtruck_DNSv3_20250405_22_16_20_SlimMPC_20250701_212315/jet_2Dtruck_SlimMPC_20250701_212315_4999_no_fields.hdf5']

    for file_path_str in filepaths:
        inspect_h5_file(Path(basedir + file_path_str))