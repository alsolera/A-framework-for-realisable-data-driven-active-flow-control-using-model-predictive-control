# compare_mpc_delay.py
#
# Compares closed-loop MPC performance across runs with different actuator
# delay levels.  Loads HDF5 result files produced by MPC_onFOM.py and produces:
#   - Time series plot: Cd, Cl, control action for each run
#   - Summary bar chart: mean Cd and mean |Cl| per delay level
#   - Summary table printed to console
#
# Usage: set RESULT_FILES below, then run from the project root.

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
# Configuration — add or remove entries as needed
# Each entry: path to the HDF5 result file produced by MPC_onFOM.py
# The actuator delay is read automatically from attrs['actuator_delay'].
# If the file predates the attribute (baseline run), use DELAY_OVERRIDES.
# ---------------------------------------------------------------------------
RESULT_FILES = [
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5",  # d=0 (baseline)
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_delay2_20260403_202409/20250906_19_38_58_SlimMPC_delay2_20260403_202409_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_delay4_20260403_181901/20250906_19_38_58_SlimMPC_delay4_20260403_181901_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_delay8_20260403_193711/20250906_19_38_58_SlimMPC_delay8_20260403_193711_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_delay12_20260403_234437/20250906_19_38_58_SlimMPC_delay12_20260403_234437_MPC_results.h5",
]

# Optional: override delay for files that lack the HDF5 attribute
# Key: filename stem (without extension), Value: int delay
DELAY_OVERRIDES = {
    # "20250906_19_38_58_SlimMPC_20260328_213407_MPC_results": 0,
}

# Output directory for plots
OUTPUT_DIR = Path("04_Results/delay_closedloop")

# Transient to skip at the start of each run (steps) for mean Cd calculation
TRANSIENT_STEPS = 200

# Control timestep expressed in convective time units (t_c = D / U_inf)
# Used to convert delay steps to physical time in the summary table.
DELTA_T_CONTROL_TC = 0.2  # Δt_control / t_c
# ---------------------------------------------------------------------------


def load_run(filepath: Path):
    """Load a single MPC result HDF5 file. Returns a dict of arrays + metadata."""
    with h5py.File(filepath, 'r') as f:
        data = {}
        for key in ['actual_forces', 'actions_applied_unscaled', 'rewards',
                    'optimal_costs', 'optimization_times']:
            if key in f:
                data[key] = f[key][:]

        # Read actuator delay from attribute, fallback to override or 0
        stem = filepath.stem
        if 'actuator_delay' in f.attrs:
            data['delay'] = int(f.attrs['actuator_delay'])
        elif stem in DELAY_OVERRIDES:
            data['delay'] = DELAY_OVERRIDES[stem]
        else:
            print(f"Warning: no actuator_delay found for {filepath.name}, assuming 0")
            data['delay'] = 0

        # Also read noise sigma if present (for completeness in table)
        data['noise_sigma'] = float(f.attrs.get('sensor_noise_sigma', 0.0))

    data['filepath'] = filepath
    n = len(data.get('actual_forces', []))
    data['n_steps'] = n
    print(f"Loaded: {filepath.name}  |  delay={data['delay']}  |  steps={n}")
    return data


def compute_summary(data: dict, transient: int = TRANSIENT_STEPS):
    """Compute steady-state mean Cd, mean |Cl|, and mean reward."""
    forces = data.get('actual_forces', np.full((1, 2), np.nan))
    start  = min(transient, len(forces) - 1)
    cd     = forces[start:, 0]
    cl     = forces[start:, 1]
    return {
        'mean_cd':     float(np.nanmean(cd)),
        'std_cd':      float(np.nanstd(cd)),
        'mean_abs_cl': float(np.nanmean(np.abs(cl))),
        'mean_reward': float(np.nanmean(data.get('rewards', [np.nan])[start:])),
    }


def plot_timeseries(runs: list, output_path: Path):
    """
    Three-panel time series: Cd, Cl, control action.
    One line per run, coloured by delay level.
    """
    cmap   = plt.cm.plasma
    delays = [r['delay'] for r in runs]
    vmin, vmax = 0, max(delays) if max(delays) > 0 else 1
    norm   = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for run in runs:
        forces  = run.get('actual_forces')
        actions = run.get('actions_applied_unscaled')
        d       = run['delay']
        color   = cmap(norm(d))
        label   = r'$d=0$ (baseline)' if d == 0 else f'$d={d}$'
        lw      = 1.8 if d == 0 else 1.2
        ls      = '-'  if d == 0 else '--'
        steps   = np.arange(len(forces))

        if forces is not None:
            axes[0].plot(steps, forces[:, 0], color=color, lw=lw, ls=ls, label=label)
            axes[1].plot(steps, forces[:, 1], color=color, lw=lw, ls=ls)
        if actions is not None:
            axes[2].plot(np.arange(len(actions)), actions, color=color, lw=lw, ls=ls)

    axes[0].set_ylabel(r'$C_d$')
    axes[1].set_ylabel(r'$C_l$')
    axes[2].set_ylabel(r'Control action')
    axes[2].set_xlabel(r'Control step')

    for ax in axes:
        ax.grid(axis='y', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shade transient region
    for ax in axes:
        ax.axvspan(0, TRANSIENT_STEPS, alpha=0.2, color='gray',
                   label='Transient' if ax is axes[0] else None)

    axes[0].set_xlim(left=0)
    axes[0].legend(loc='upper right', fontsize='small', frameon=False, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved time series plot to {output_path}")
    plt.close(fig)


def plot_summary(runs: list, summaries: list, output_path: Path):
    """
    Two-panel bar chart: mean Cd and mean |Cl| vs actuator delay.
    Error bars show +/- std Cd.
    """
    delays    = np.array([r['delay'] for r in runs])
    mean_cds  = np.array([s['mean_cd']      for s in summaries])
    std_cds   = np.array([s['std_cd']       for s in summaries])
    mean_cls  = np.array([s['mean_abs_cl']  for s in summaries])
    labels    = [f'$d={d}$' for d in delays]

    x      = np.arange(len(runs))
    width  = 0.6
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(runs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    ax1.bar(x, mean_cds, width, yerr=std_cds, color=colors,
            capsize=4, error_kw=dict(linewidth=1.0))
    # Reference line: uncontrolled Cd
    ax1.axhline(1.051, color='k', linestyle=':', linewidth=1.0)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize='small')
    ax1.set_xlabel(r'Actuator delay $d$ (steps)')
    ax1.set_ylabel(r'Mean $C_d$')
    ax1.text(0, 1.051, 'Uncontrolled', fontsize='small',
             horizontalalignment='left', verticalalignment='bottom')
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, size='large')

    ax2.bar(x, mean_cls, width, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize='small')
    ax2.set_xlabel(r'Actuator delay $d$ (steps)')
    ax2.set_ylabel(r'Mean $|C_l|$')
    ax2.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, size='large')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved summary bar chart to {output_path}")
    plt.close(fig)


def print_table(runs: list, summaries: list):
    """Print a summary table to console."""
    header = (f"{'Delay (steps)':>14} {'Delay (t_c)':>12} {'Mean Cd':>10} "
              f"{'Std Cd':>10} {'Mean |Cl|':>12} {'Mean reward':>13} {'Steps':>7}")
    print("\n" + "=" * len(header))
    print("Closed-loop MPC actuator delay robustness summary")
    print(f"(Steady-state: steps {TRANSIENT_STEPS}+, "
          f"dt_control = {DELTA_T_CONTROL_TC:.2f} t_c)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for run, s in zip(runs, summaries):
        d = run['delay']
        delay_tc = d * DELTA_T_CONTROL_TC
        label = f"{d} (baseline)" if d == 0 else f"{d}"
        print(f"{label:>14} {delay_tc:>12.2f} {s['mean_cd']:>10.4f} {s['std_cd']:>10.4f} "
              f"{s['mean_abs_cl']:>12.4f} {s['mean_reward']:>13.4f} "
              f"{run['n_steps']:>7}")
    print("=" * len(header))

    # Drag reduction relative to uncontrolled
    print("\nDrag reduction vs uncontrolled (Cd_ref = 1.051):")
    for run, s in zip(runs, summaries):
        reduction = (1.051 - s['mean_cd']) / 1.051 * 100
        print(f"  d={run['delay']}:  {reduction:+.2f}%")

    # Drag reduction degradation relative to baseline (d=0) if available
    baseline_runs = [r for r in runs if r['delay'] == 0]
    if baseline_runs:
        baseline_cd = [s['mean_cd'] for r, s in zip(runs, summaries) if r['delay'] == 0][0]
        print(f"\nDrag reduction loss vs baseline (d=0, mean Cd = {baseline_cd:.4f}):")
        for run, s in zip(runs, summaries):
            if run['delay'] == 0:
                continue
            loss = (s['mean_cd'] - baseline_cd) / baseline_cd * 100
            print(f"  d={run['delay']}:  {loss:+.2f}% increase in mean Cd")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load all runs ---
    runs = []
    for path_str in RESULT_FILES:
        p = Path(path_str)
        if not p.exists():
            print(f"WARNING: file not found, skipping: {p}")
            continue
        runs.append(load_run(p))

    if not runs:
        print("No valid result files found. Check RESULT_FILES.")
        return

    # Sort by delay for consistent ordering
    runs.sort(key=lambda r: r['delay'])

    # --- Compute summaries ---
    summaries = [compute_summary(r) for r in runs]

    # --- Print table ---
    print_table(runs, summaries)

    # --- Plots ---
    plot_timeseries(runs, OUTPUT_DIR / "mpc_delay_timeseries.png")
    plot_summary(runs,    summaries, OUTPUT_DIR / "mpc_delay_summary.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()