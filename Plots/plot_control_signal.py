import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pywt # Using PyWavelets for Continuous Wavelet Transform
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def plot_signal_and_spectrogram(csv_filepath, max_idx=5000):
    """
    Reads a time-series from a CSV file, plots the signal and its wavelet scalogram,
    and creates a clean, publication-ready plot.
    """
    filepath = Path(csv_filepath)
    if not filepath.is_file():
        print(f"Error: Data file not found at '{filepath}'")
        print("Please make sure the CSV_FILE_PATH variable is set correctly.")
        return

    # Read the data from the CSV file
    try:
        # Assume no header and assign column names directly for robustness
        df = pd.read_csv(filepath, comment='#', skipinitialspace=True, header=None, names=['time', 'value'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Verify that the necessary columns exist
    if 'time' not in df.columns or 'value' not in df.columns:
        print(f"Error: CSV file '{filepath}' must contain 'time' and 'value' columns.")
        return

    # Prepare data for plotting and processing
    time = df['time'][:max_idx].to_numpy()
    value = df['value'][:max_idx].to_numpy()
    value = value / max(value)

    # --- Wavelet Transform Calculation ---
    if len(time) < 2:
        print("Error: Not enough data points for spectrogram.")
        return
    dt = time[1] - time[0]
    if dt <= 0:
        print(f"Error: Time data is not monotonically increasing. dt = {dt}")
        return

    wavelet = 'cmor1.5-1.0'  # A common choice: Complex Morlet wavelet
    # Define the scales for the CWT. A lower scale corresponds to a higher frequency.
    total_scales = 256
    scales = np.arange(1, total_scales)

    # Perform the Continuous Wavelet Transform
    coefficients, frequencies = pywt.cwt(value, scales, wavelet, dt)

    # --- Create the Plot ---
    # Two subplots, shared x-axis, equal height.
    fig, axs = plt.subplots(
        2, 1,
        figsize=(8, 4),
        sharex=True,
        constrained_layout=True  # Use a better layout manager
    )

    # --- Top Plot: Time-domain signal ---
    axs[0].plot(time, value, linewidth=0.5)
    axs[0].set_ylabel('Normalised control')
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_ylim(-1.01, 1.01)
    axs[0].set_xlim(time[0], time[-1]+1)
    # --- Remove visual clutter ---
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].get_xaxis().set_visible(False)

    # --- Bottom Plot: Scalogram (Wavelet Spectrogram) ---
    # Calculate power from the absolute value of the complex coefficients
    power = np.abs(coefficients)**2
    power_db = 10 * np.log10(power + 1e-12)
    pcm = axs[1].pcolormesh(time, frequencies, power_db, cmap='viridis', shading='gouraud', rasterized=True,
                            vmin=-80, vmax=10)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Frequency')
    axs[1].set_xlabel('$t_c$')
    # --- Remove visual clutter ---
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # Add a colorbar for the spectrogram, associating it with both axes to keep them aligned
    fig.colorbar(pcm, ax=axs.tolist(), label='Power (dB)', pad=0.01)

    # --- Save the Figure ---
    output_dir = Path("../04_Results/paper_figures/")
    output_filename_base = filepath.stem
    output_path = output_dir / (output_filename_base + '_wavelet.png') # Changed filename

    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.savefig(output_path.with_suffix('.eps'))

    print(f"Successfully saved plot to {output_path} (and .pdf/.eps versions)")
    plt.close(fig)


if __name__ == "__main__":
    # --- USER: Please set the path to your CSV file here ---
    CSV_FILE_PATH = '../gymprecice-run/jet_2Dtruck_DNSv3_20250307_2155_FMsignal/control_inputs.csv'

    plot_signal_and_spectrogram(CSV_FILE_PATH)