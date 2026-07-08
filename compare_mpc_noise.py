# compare_mpc_noise.py
#
# Compares closed-loop MPC performance across runs with different sensor noise levels.
# Loads HDF5 result files produced by MPC_onFOM.py and produces:
#   - Time series plot: Cd, Cl, control action for each run
#   - Summary bar chart: mean Cd and mean |Cl| per noise level
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
# Configuration — add or remove entries as needed (up to ~5 runs recommended)
# Each entry: path to the HDF5 result file produced by MPC_onFOM.py
# The noise level is read automatically from the file's attrs['sensor_noise_sigma'].
# If the file predates the noise attr (clean run without our changes), set
# sensor_noise_sigma manually via the override dict below.
# ---------------------------------------------------------------------------
RESULT_FILES = [
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5",  # clean
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise10pct_20260331_213408/20250906_19_38_58_SlimMPC_noise10pct_20260331_213408_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise20pct_20260331_233643/20250906_19_38_58_SlimMPC_noise20pct_20260331_233643_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise30pct_20260401_094715/20250906_19_38_58_SlimMPC_noise30pct_20260401_094715_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise50pct_20260401_104044/20250906_19_38_58_SlimMPC_noise50pct_20260401_104044_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise70pct_20260401_183401/20250906_19_38_58_SlimMPC_noise70pct_20260401_183401_MPC_results.h5",
    "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_noise90pct_20260402_120440/20250906_19_38_58_SlimMPC_noise90pct_20260402_120440_MPC_results.h5",
]

# Optional: override noise sigma label for files that lack the HDF5 attribute
# Key: filename stem (without extension), Value: float sigma
SIGMA_OVERRIDES = {
    # "20250906_19_38_58_SlimMPC_20260331_XXXXXX_MPC_results": 0.0,
}

# Output directory for plots
OUTPUT_DIR = Path("04_Results/noise_closedloop")

# Transient to skip at the start of each run (steps) for mean Cd calculation
TRANSIENT_STEPS = 200
# ---------------------------------------------------------------------------


def load_run(filepath: Path):
    """Load a single MPC result HDF5 file. Returns a dict of arrays + metadata."""
    with h5py.File(filepath, 'r') as f:
        data = {}
        for key in ['actual_forces', 'actions_applied_unscaled', 'rewards',
                    'optimal_costs', 'optimization_times']:
            if key in f:
                data[key] = f[key][:]

        # Read noise sigma from attribute, fallback to override or 0.0
        stem = filepath.stem
        if 'sensor_noise_sigma' in f.attrs:
            data['noise_sigma'] = float(f.attrs['sensor_noise_sigma'])
        elif stem in SIGMA_OVERRIDES:
            data['noise_sigma'] = SIGMA_OVERRIDES[stem]
        else:
            print(f"Warning: no noise_sigma found for {filepath.name}, assuming 0.0")
            data['noise_sigma'] = 0.0

    data['filepath'] = filepath
    n = len(data.get('actual_forces', []))
    data['n_steps'] = n
    print(f"Loaded: {filepath.name}  |  noise_sigma={data['noise_sigma']:.3f}  |  steps={n}")
    return data


def compute_summary(data: dict, transient: int = TRANSIENT_STEPS):
    """Compute steady-state mean Cd, mean |Cl|, and mean reward."""
    forces = data.get('actual_forces', np.full((1, 2), np.nan))
    start  = min(transient, len(forces) - 1)
    cd     = forces[start:, 0]
    cl     = forces[start:, 1]
    return {
        'mean_cd':    float(np.nanmean(cd)),
        'std_cd':     float(np.nanstd(cd)),
        'mean_abs_cl':float(np.nanmean(np.abs(cl))),
        'mean_reward':float(np.nanmean(data.get('rewards', [np.nan])[start:])),
    }


def plot_timeseries(runs: list, output_path: Path):
    """
    Three-panel time series: Cd, Cl, control action.
    One line per run, coloured by noise level.
    """
    cmap   = plt.cm.viridis
    sigmas = [r['noise_sigma'] for r in runs]
    vmin, vmax = 0.0, max(sigmas) if max(sigmas) > 0 else 1.0
    norm   = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for run in runs:
        forces  = run.get('actual_forces')
        actions = run.get('actions_applied_unscaled')
        sigma   = run['noise_sigma']
        color   = cmap(norm(sigma))
        label   = r'$\sigma=0$ (clean)' if sigma == 0.0 else f'$\\sigma={sigma*100:.0f}\\%$'
        lw      = 1.8 if sigma == 0.0 else 1.2
        ls      = '-'  if sigma == 0.0 else '--'
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
    Two-panel bar chart: mean Cd and mean |Cl| vs noise level.
    Error bars show ± std Cd.
    """
    sigmas    = np.array([r['noise_sigma'] * 100 for r in runs])   # in %
    mean_cds  = np.array([s['mean_cd']      for s in summaries])
    std_cds   = np.array([s['std_cd']       for s in summaries])
    mean_cls  = np.array([s['mean_abs_cl']  for s in summaries])
    labels    = [r'Clean' if s == 0 else f'{s:.0f}\%' for s in sigmas]

    x      = np.arange(len(runs))
    width  = 0.6
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    bars1 = ax1.bar(x, mean_cds, width, yerr=std_cds, color=colors,
                    capsize=4, error_kw=dict(linewidth=1.0))
    # Reference line: uncontrolled Cd
    ax1.axhline(1.051, color='k', linestyle=':', linewidth=1.0)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize='small')
    ax1.set_xlabel(r'Noise $\sigma$ (\% of global $\sigma$)')
    ax1.set_ylabel(r'Mean $C_d$')
    ax1.text(0, 1.051, 'Uncontrolled', fontsize='small', horizontalalignment='left',
     verticalalignment='bottom')
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, size='large')

    bars2 = ax2.bar(x, mean_cls, width, color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize='small')
    ax2.set_xlabel(r'Noise $\sigma$ (\% of global $\sigma$)')
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
    header = f"{'Noise sigma':>14} {'Mean Cd':>10} {'Std Cd':>10} {'Mean |Cl|':>12} {'Mean reward':>13} {'Steps':>7}"
    print("\n" + "="*len(header))
    print("Closed-loop MPC noise robustness summary")
    print(f"(Steady-state: steps {TRANSIENT_STEPS}+)")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for run, s in zip(runs, summaries):
        sigma = run['noise_sigma']
        label = f"{sigma*100:.1f}% (clean)" if sigma == 0.0 else f"{sigma*100:.1f}%"
        print(f"{label:>14} {s['mean_cd']:>10.4f} {s['std_cd']:>10.4f} "
              f"{s['mean_abs_cl']:>12.4f} {s['mean_reward']:>13.4f} "
              f"{run['n_steps']:>7}")
    print("="*len(header))
    # Drag reduction relative to uncontrolled
    print("\nDrag reduction vs uncontrolled (Cd_ref = 1.051):")
    for run, s in zip(runs, summaries):
        reduction = (1.051 - s['mean_cd']) / 1.051 * 100
        sigma_label = "clean" if run['noise_sigma'] == 0.0 else f"{run['noise_sigma']*100:.1f}%"
        print(f"  sigma={sigma_label:>8s}:  {reduction:+.2f}%")


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

    # Sort by noise sigma for consistent ordering
    runs.sort(key=lambda r: r['noise_sigma'])

    # --- Compute summaries ---
    summaries = [compute_summary(r) for r in runs]

    # --- Print table ---
    print_table(runs, summaries)

    # --- Plots ---
    plot_timeseries(runs, OUTPUT_DIR / "mpc_noise_timeseries.png")
    plot_summary(runs,    summaries, OUTPUT_DIR / "mpc_noise_summary.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()