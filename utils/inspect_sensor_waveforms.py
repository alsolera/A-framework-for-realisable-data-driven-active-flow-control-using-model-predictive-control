# inspect_sensor_waveforms.py  (temporary diagnostic script)
#
# Plots the raw sensor waveforms for the 4 selected sensors across MPC runs
# with different noise levels. Useful to visually confirm noise is present
# and to assess its magnitude relative to the signal.
#
# Loads: observations_history_unscaled  (shape: n_steps, lookback, n_sensors)
#        sensor_noise_sigma attribute
#
# Each step's observation is the lookback window at that instant.
# We reconstruct the continuous sensor time series by taking obs[t, -1, :]
# (the most recent reading at each control step).

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULT_FILES = [
    "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5",  # clean
    # "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise10pct_20260331_213408/20250906_19_38_58_SlimMPC_noise10pct_20260331_213408_MPC_results.h5",
    # "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise20pct_20260331_233643/20250906_19_38_58_SlimMPC_noise20pct_20260331_233643_MPC_results.h5",
    # "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise30pct_20260401_094715/20250906_19_38_58_SlimMPC_noise30pct_20260401_094715_MPC_results.h5",
    # "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise50pct_20260401_104044/20250906_19_38_58_SlimMPC_noise50pct_20260401_104044_MPC_results.h5",
    # "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise70pct_20260401_183401/20250906_19_38_58_SlimMPC_noise70pct_20260401_183401_MPC_results.h5",
    "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise90pct_20260402_120440/20250906_19_38_58_SlimMPC_noise90pct_20260402_120440_MPC_results.h5",
]

# Indices of the 4 selected sensors (must match what was used in the MPC run)
SELECTED_INDICES = [39, 79, 80, 89]   # adjust if different

# How many control steps to plot (None = all)
N_STEPS_TO_PLOT = 200

# Output path
OUTPUT_PATH = Path("../04_Results/noise_closedloop/sensor_waveforms.png")
# ---------------------------------------------------------------------------


def load_sensors(filepath: Path):
    """Extract the continuous sensor time series from observations_history_unscaled."""
    with h5py.File(filepath, 'r') as f:
        if 'observations_history_unscaled' not in f:
            raise KeyError(f"'observations_history_unscaled' not found in {filepath}")
        obs = f['observations_history_unscaled'][:]   # (n_steps, lookback, n_sensors)
        sigma = float(f.attrs.get('sensor_noise_sigma', 0.0))

    # Take the most recent timestep from each window -> (n_steps, n_sensors)
    sensors = obs[:, -1, :]
    return sensors, sigma


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for path_str in RESULT_FILES:
        p = Path(path_str)
        if not p.exists():
            print(f"WARNING: not found, skipping: {p}")
            continue
        sensors, sigma = load_sensors(p)
        runs.append({'sensors': sensors, 'sigma': sigma, 'path': p})
        print(f"Loaded {p.name}: shape={sensors.shape}, noise_sigma={sigma:.3f}")

    if not runs:
        print("No files loaded."); return

    runs.sort(key=lambda r: r['sigma'])

    n_sensors = len(SELECTED_INDICES)
    n_runs    = len(runs)

    # Colour map: one colour per run
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, n_runs))

    # --- Figure 1: overlay all runs for each selected sensor ---
    fig1, axes = plt.subplots(n_sensors, 1, figsize=(8, 1.5 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for ax_idx, sensor_idx in enumerate(SELECTED_INDICES):
        ax = axes[ax_idx]
        for run_idx, run in enumerate(runs):
            s     = run['sensors']
            sigma = run['sigma']
            n     = N_STEPS_TO_PLOT if N_STEPS_TO_PLOT else len(s)
            n     = min(n, len(s))
            steps = np.arange(n)
            label = r'Clean' if sigma == 0.0 else f'$\\sigma={sigma*100:.0f}\\%$'
            lw    = 1.5 if sigma == 0.0 else 0.9
            ls    = '-' if sigma == 0.0 else '--'
            ax.plot(steps, s[:n, sensor_idx], color=colors[run_idx],
                    lw=lw, ls=ls, label=label, alpha=0.9)

        ax.set_ylabel(f'Sensor {sensor_idx}', fontsize='small')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Control step')
    handles, labels = axes[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01),
                ncol=n_runs, frameon=False, fontsize='small')
    fig1.suptitle('Selected sensor waveforms (unscaled)', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PATH.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved overlay plot to {OUTPUT_PATH}")
    plt.close(fig1)

    # --- Figure 2: zoom into a short window to clearly see noise ---
    zoom_steps = min(50, N_STEPS_TO_PLOT or 50,
                     min(len(r['sensors']) for r in runs))
    fig2, axes2 = plt.subplots(n_sensors, 1, figsize=(8, 1.5 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes2 = [axes2]

    for ax_idx, sensor_idx in enumerate(SELECTED_INDICES):
        ax = axes2[ax_idx]
        for run_idx, run in enumerate(runs):
            s     = run['sensors']
            sigma = run['sigma']
            steps = np.arange(zoom_steps)
            label = r'Clean' if sigma == 0.0 else f'$\\sigma={sigma*100:.0f}\\%$'
            lw    = 1.8 if sigma == 0.0 else 1.0
            ls    = '-' if sigma == 0.0 else '--'
            ax.plot(steps, s[:zoom_steps, sensor_idx], color=colors[run_idx],
                    lw=lw, ls=ls, label=label, alpha=0.9)

        ax.set_ylabel(f'Sensor {sensor_idx}', fontsize='small')
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes2[-1].set_xlabel('Control step')
    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, 1.01),
                ncol=n_runs, frameon=False, fontsize='small')
    fig2.suptitle(f'Sensor waveforms — first {zoom_steps} steps (zoom)', y=1.03)
    plt.tight_layout()
    zoom_path = OUTPUT_PATH.parent / (OUTPUT_PATH.stem + '_zoom.png')
    plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
    plt.savefig(zoom_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved zoom plot to {zoom_path}")
    plt.close(fig2)

    # --- Console stats ---
    print("\nSensor signal statistics (unscaled, selected sensors only):")
    print(f"{'Run':>30}  {'sigma':>6}  {'signal std':>12}  {'noise/signal':>14}")
    print("-" * 68)
    clean_stds = None
    for run in runs:
        s_sel = run['sensors'][:, SELECTED_INDICES]
        stds  = np.std(s_sel, axis=0)
        sigma = run['sigma']
        name  = Path(run['path']).name[:30]
        if sigma == 0.0:
            clean_stds = stds
            print(f"{name:>30}  {sigma:>6.3f}  {stds.mean():>12.5f}  {'(reference)':>14}")
        else:
            snr = sigma * stds.mean() / stds.mean() if clean_stds is None else sigma
            print(f"{name:>30}  {sigma:>6.3f}  {stds.mean():>12.5f}  {sigma:>13.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()