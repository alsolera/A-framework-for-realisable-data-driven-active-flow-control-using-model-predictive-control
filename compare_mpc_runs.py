# compare_mpc_runs.py
#
# Compares two (or more) MPC-on-FOM runs from their saved *_MPC_results.h5
# files.  Designed to show that different optimizer learning rates yield
# equivalent closed-loop drag reduction despite differences in the control
# signal smoothness.
#
# Outputs (saved to OUTPUT_DIR):
#   - mpc_comparison_forces.pdf/png/eps    -- Cd and Cl time histories
#   - mpc_comparison_actions.pdf/png/eps   -- control signal overlay
#   - mpc_comparison_summary.pdf/png/eps   -- combined 3-panel figure
#   - mpc_comparison_metrics.json          -- summary statistics

import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Each entry: (path_to_h5, label_for_legend, colour)
RUNS = [
    {
        'h5':    'gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260404_085259/20250906_19_38_58_SlimMPC_20260404_085259_MPC_results.h5',
        'label': r'LR=$10^{-2}$',
        'color': 'tab:orange',
    },
    {
        'h5':    'gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5',
        'label': r'LR=$10^{-3}$',
        'color': 'black',
    },
    {
        'h5':    'gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260404_100340/20250906_19_38_58_SlimMPC_20260404_100340_MPC_results.h5',
        'label': r'LR=$10^{-4}$',
        'color': 'tab:blue',
    },
]

# Time axis
DELTA_T_CONTROL = 0.2          # control time step (same as MPC_onFOM.py)

# Uncontrolled baseline Cd (for reference line)
CD_UNCONTROLLED = 1.051

OUTPUT_DIR = Path('04_Results/lr_comparison')
# ---------------------------------------------------------------------------


def load_run(h5_path):
    """Load relevant fields from an MPC_results.h5 file."""
    h5_path = Path(h5_path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"MPC results file not found: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        data = {}
        data['actions'] = f['actions_applied_unscaled'][:] if 'actions_applied_unscaled' in f else None
        data['forces']  = f['actual_forces'][:]            if 'actual_forces' in f else None
        data['costs']   = f['optimal_costs'][:]            if 'optimal_costs' in f else None
        data['rewards'] = f['rewards'][:]                  if 'rewards' in f else None
        data['opt_times'] = f['optimization_times'][:]     if 'optimization_times' in f else None
        # Store metadata
        data['sensor_noise_sigma'] = f.attrs.get('sensor_noise_sigma', 0.0)
        data['actuator_delay']     = f.attrs.get('actuator_delay', 0)
    print(f"Loaded {h5_path.name}: {len(data['actions'])} steps")
    return data


def build_time_axis(n_steps, dt_control):
    """Build time vectors."""
    t = np.arange(n_steps) * dt_control
    return t


def compute_metrics(data, label, transient_steps=50):
    """Compute summary statistics after discarding initial transient."""
    forces = data['forces']
    actions = data['actions']
    n = min(len(forces), len(actions))
    s = min(transient_steps, n // 2)  # start of steady-state window

    cd = forces[s:n, 0]
    cl = forces[s:n, 1]
    u  = actions[s:n] if actions.ndim == 1 else actions[s:n, 0]

    metrics = {
        'label':       label,
        'n_steps':     n,
        'transient':   s,
        'mean_cd':     float(np.mean(cd)),
        'std_cd':      float(np.std(cd)),
        'mean_cl':     float(np.mean(np.abs(cl))),
        'std_cl':      float(np.std(cl)),
        'mean_action': float(np.mean(u)),
        'std_action':  float(np.std(u)),
        'action_rate': float(np.mean(np.abs(np.diff(u)))),
    }
    return metrics


def plot_summary(runs_data, output_dir):
    """Three-panel figure: Cd, Cl, action signal."""
    n_runs = len(runs_data)

    fig, axes = plt.subplots(3, 1, figsize=(7, 4), sharex=True)

    for rd in runs_data:
        data  = rd['data']
        label = rd['label']
        color = rd['color']

        forces  = data['forces']
        actions = data['actions']
        n = min(len(forces), len(actions))

        t = build_time_axis(n, DELTA_T_CONTROL)
        cd = forces[:n, 0]
        cl = forces[:n, 1]
        u  = actions[:n] if actions.ndim == 1 else actions[:n, 0]

        axes[0].plot(t, cd, color=color, linewidth=0.6, label=label, alpha=0.85)
        axes[1].plot(t, cl, color=color, linewidth=0.6, label=label, alpha=0.85)
        axes[2].plot(t, u,  color=color, linewidth=0.6, label=label, alpha=0.85)

    # -- Cd panel --
    ax = axes[0]
    ax.axhline(CD_UNCONTROLLED, color='grey', linestyle='--', linewidth=0.8,
               label=r'Uncontrolled $\bar{C}_d$')
    ax.set_ylabel(r'$C_d$')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.text(-0.09, 1.05, '(a)', transform=ax.transAxes, size='large')

    # -- Cl panel --
    ax = axes[1]
    ax.set_ylabel(r'$C_l$')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.text(-0.09, 1.05, '(b)', transform=ax.transAxes, size='large')

    # -- Action panel --
    ax = axes[2]
    ax.set_ylabel(r'Action $a$')
    xlabel = r'$t_c$'
    ax.set_xlabel(xlabel)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.text(-0.09, 1.05, '(c)', transform=ax.transAxes, size='large')

    plt.tight_layout()
    fpath = output_dir / 'mpc_comparison_summary'
    for ext in ['.pdf', '.png']:
        fig.savefig(fpath.with_suffix(ext), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary figure -> {fpath}.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load all runs --------------------------------------------------------
    runs_data = []
    for run_cfg in RUNS:
        data = load_run(run_cfg['h5'])
        runs_data.append({
            'data':  data,
            'label': run_cfg['label'],
            'color': run_cfg['color'],
        })

    # -- Compute and print summary metrics ------------------------------------
    all_metrics = []
    print("\n" + "=" * 72)
    print(f"  {'Label':<20s}  {'mean Cd':>8s}  {'std Cd':>8s}  "
          f"{'mean |Cl|':>9s}  {'action rate':>12s}")
    print("-" * 72)
    for rd in runs_data:
        m = compute_metrics(rd['data'], rd['label'])
        all_metrics.append(m)
        print(f"  {m['label']:<20s}  {m['mean_cd']:>8.4f}  {m['std_cd']:>8.4f}  "
              f"{m['mean_cl']:>9.4f}  {m['action_rate']:>12.6f}")
    print("=" * 72)

    # -- Save metrics to JSON -------------------------------------------------
    metrics_file = OUTPUT_DIR / 'mpc_comparison_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved -> {metrics_file}")

    # -- Plot -----------------------------------------------------------------
    plot_summary(runs_data, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()