"""
mpc_lr_sensitivity.py -- Offline sensitivity analysis for MPC internal optimizer.

Addresses Reviewer 2.2:
  (A) Learning-rate sweep: how sensitive is the converged cost / optimal action
      to the Adam LR used inside the MPC?
  (B) Multi-initialization test: starting from random initial guesses instead of
      warm-start, do all seeds converge to the same solution?  If yes -> no
      problematic local minima.

Usage:
    python mpc_lr_sensitivity.py

All heavy parameters are collected in the CONFIG dict at the bottom.
The script loads a previously saved MPC_results.h5 (from MPC_onFOM.py) and
re-runs controller.optimize() offline -- no CFD needed.
"""

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
from pathlib import Path
import time

# -- project imports ----------------------
from libs.MPC import MPCControllerLatent


# ===============================================================================
#  Low-level optimize that accepts an explicit initial guess & LR
# ===============================================================================
def optimize_from_guess(
    controller: MPCControllerLatent,
    past_sensors_unscaled: np.ndarray,   # (lookback, n_sensors)
    last_applied_action_unscaled: np.ndarray,  # (action_dim,)
    initial_guess: np.ndarray,           # (horizon, action_dim)
    lr: float,
    num_optim_steps: int,
    return_cost_trace: bool = False,
    cancel_dropout: bool = False,
):
    """
    Run the MPC internal optimization with a *given* initial guess and LR.

    Returns
    -------
    converged_cost : float
    optimal_actions : np.ndarray, shape (horizon, action_dim)
    predicted_cd    : np.ndarray, shape (horizon,)
    cost_trace      : list[float]  (only if return_cost_trace=True)
    """
    device = controller.device
    H = controller.horizon

    # -- encode sensor history ------------------------------------------------
    past_sensors_tensor_full = torch.tensor(
        past_sensors_unscaled, dtype=torch.float32, device=device
    )
    if controller.use_slim_encoder and controller.selected_indices is not None:
        past_sensors_tensor = past_sensors_tensor_full[:, controller.selected_indices]
    else:
        past_sensors_tensor = past_sensors_tensor_full

    past_sensors_scaled = (
        (past_sensors_tensor - controller.p_mean_tensor) / controller.p_std_tensor
    ).unsqueeze(0)

    last_action_tensor = torch.tensor(
        last_applied_action_unscaled, dtype=torch.float32, device=device
    )

    # -- initial guess -> optimizable tensor -----------------------------------
    u_tensor = torch.tensor(
        initial_guess.copy(), dtype=torch.float32, device=device, requires_grad=True
    )
    with torch.no_grad():
        u_tensor.clamp_(controller.limits[0], controller.limits[1])

    optimizer = torch.optim.Adam([u_tensor], lr=lr)

    # Put models in train mode (enables dropout noise -- same as production MPC)
    controller.encoder.train()
    controller.dynamics_model.train()
    controller.force_decoder.train()
    
    # Save original dropout rates and set to 0
    if cancel_dropout:
        _orig_drop_p = []
        for m in controller.force_decoder.modules():
            if isinstance(m, torch.nn.Dropout):
                _orig_drop_p.append(m.p)
                m.p = 0.0

    cost_trace = []
    for _ in range(num_optim_steps):
        optimizer.zero_grad()
        pred_forces_scaled = controller.predict_horizon_torch(past_sensors_scaled, u_tensor)
        cost = controller.cost_function_torch(pred_forces_scaled, u_tensor, last_action_tensor)
        cost.backward()
        optimizer.step()
        with torch.no_grad():
            u_tensor.clamp_(controller.limits[0], controller.limits[1])
        cost_trace.append(cost.item())
        
    # Restore original dropout rates
    if cancel_dropout:
        idx = 0
        for m in controller.force_decoder.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = _orig_drop_p[idx]
                idx += 1

    controller.encoder.eval()
    controller.dynamics_model.eval()
    controller.force_decoder.eval()

    # -- extract results ------------------------------------------------------
    with torch.no_grad():
        final_forces_scaled = controller.predict_horizon_torch(past_sensors_scaled, u_tensor)
        final_cost = controller.cost_function_torch(
            final_forces_scaled, u_tensor, last_action_tensor
        )
        final_forces_unscaled = (
            final_forces_scaled * controller.forces_std_tensor + controller.forces_mean_tensor
        )
        predicted_cd = final_forces_unscaled[:, 0].cpu().numpy()
        optimal_actions = u_tensor.detach().cpu().numpy()

    out = (final_cost.item(), optimal_actions, predicted_cd)
    if return_cost_trace:
        out = out + (cost_trace,)
    return out


# ===============================================================================
#  Snapshot loader
# ===============================================================================
def load_snapshots(h5_path: str, n_snapshots: int, lookback: int):
    """
    Load evenly-spaced snapshots from a previous MPC run.

    Returns
    -------
    snapshots : list of dict, each with keys:
        'obs'             : (lookback, n_sensors)
        'last_action'     : (action_dim,)
        'step_idx'        : int
    """
    with h5py.File(h5_path, 'r') as f:
        obs_history = f['observations_history_unscaled'][:]    # (N, lookback, n_sensors)
        actions     = f['actions_applied_unscaled'][:]         # (N,) or (N, action_dim)

    N = obs_history.shape[0]
    if actions.ndim == 1:
        actions = actions[:, None]

    indices = np.linspace(0, N - 1, n_snapshots, dtype=int)
    # Avoid index 0 where last_action is zero (transient) -- shift to at least 1
    indices = np.clip(indices, 1, N - 1)
    indices = np.unique(indices)

    snapshots = []
    for idx in indices:
        snapshots.append({
            'obs': obs_history[idx],                   # (lookback, n_sensors)
            'last_action': actions[idx - 1],           # action applied at previous step
            'step_idx': int(idx),
        })
    print(f"Loaded {len(snapshots)} snapshots from {h5_path} (total steps: {N})")
    return snapshots


# ===============================================================================
#  Experiment A -- Learning-rate sweep
# ===============================================================================
def run_lr_sweep(controller, snapshots, lr_values, num_optim_steps, cancel_dropout=False):
    """
    For each snapshot x LR, optimize from the default warm-start guess
    (last_action repeated over horizon, mimicking cold start).
    """
    H = controller.horizon
    results = []  # list of dicts

    for snap in snapshots:
        # Build default initial guess: repeat last action over horizon
        init_guess = np.tile(snap['last_action'], (H, 1))

        for lr in lr_values:
            cost, actions, pred_cd, trace = optimize_from_guess(
                controller,
                past_sensors_unscaled=snap['obs'],
                last_applied_action_unscaled=snap['last_action'],
                initial_guess=init_guess,
                lr=lr,
                num_optim_steps=num_optim_steps,
                return_cost_trace=True,
                cancel_dropout=cancel_dropout
            )
            results.append({
                'step_idx': snap['step_idx'],
                'lr': lr,
                'cost': cost,
                'mean_cd': np.mean(pred_cd),
                'action_first': actions[0, 0],
                'cost_trace': trace,
            })
            print(f"  [LR sweep] step={snap['step_idx']:>5d}  lr={lr:.0e}  "
                  f"cost={cost:.5f}  mean_Cd={np.mean(pred_cd):.4f}")

    return results


# ===============================================================================
#  Experiment B -- Multi-initialization (local minima test)
# ===============================================================================
def run_multi_init(controller, snapshots, n_seeds, lr, num_optim_steps, rng_seed=42, cancel_dropout=False):
    """
    For each snapshot, optimize from n_seeds random initial guesses
    (uniform within actuator limits) + 1 warm-start baseline.
    """
    H = controller.horizon
    action_dim = controller.action_dim
    lo, hi = controller.limits[0], controller.limits[1]
    rng = np.random.default_rng(rng_seed)

    results = []

    for snap in snapshots:
        # -- warm-start baseline ----------------------------------------------
        init_ws = np.tile(snap['last_action'], (H, 1))
        cost_ws, actions_ws, pred_cd_ws, _ = optimize_from_guess(
            controller, snap['obs'], snap['last_action'],
            init_ws, lr, num_optim_steps, return_cost_trace=True, cancel_dropout=cancel_dropout,
        )
        results.append({
            'step_idx': snap['step_idx'],
            'seed': 'warm_start',
            'cost': cost_ws,
            'mean_cd': np.mean(pred_cd_ws),
            'action_first': actions_ws[0, 0],
        })

        # -- random seeds -----------------------------------------------------
        for s in range(n_seeds):
            init_rand = rng.uniform(lo, hi, size=(H, action_dim))
            cost_r, actions_r, pred_cd_r, _ = optimize_from_guess(
                controller, snap['obs'], snap['last_action'],
                init_rand, lr, num_optim_steps, return_cost_trace=True, cancel_dropout=cancel_dropout
            )
            results.append({
                'step_idx': snap['step_idx'],
                'seed': s,
                'cost': cost_r,
                'mean_cd': np.mean(pred_cd_r),
                'action_first': actions_r[0, 0],
            })

        # quick summary for this snapshot
        costs_here = [r['cost'] for r in results if r['step_idx'] == snap['step_idx']]
        print(f"  [Multi-init] step={snap['step_idx']:>5d}  "
              f"cost range=[{min(costs_here):.5f}, {max(costs_here):.5f}]  "
              f"std={np.std(costs_here):.6f}")

    return results


# ===============================================================================
#  Plotting
# ===============================================================================
def plot_lr_sweep(results, output_dir):
    """Box plot of converged cost and mean predicted Cd vs LR."""
    lr_vals = sorted(set(r['lr'] for r in results))
    lr_labels = [f"{lr:.0e}" for lr in lr_vals]

    # -- Collect data per LR --------------------------------------------------
    costs_by_lr = {lr: [] for lr in lr_vals}
    cd_by_lr    = {lr: [] for lr in lr_vals}
    for r in results:
        costs_by_lr[r['lr']].append(r['cost'])
        cd_by_lr[r['lr']].append(r['mean_cd'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Cost
    ax = axes[0]
    bp = ax.boxplot([costs_by_lr[lr] for lr in lr_vals], labels=lr_labels,
                    patch_artist=True, widths=0.5)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax.set_xlabel('Adam learning rate')
    ax.set_ylabel('Converged MPC cost')
    ax.set_title('(a) Optimizer cost vs. learning rate')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Mean Cd
    ax = axes[1]
    bp = ax.boxplot([cd_by_lr[lr] for lr in lr_vals], labels=lr_labels,
                    patch_artist=True, widths=0.5)
    for patch in bp['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.6)
    ax.set_xlabel('Adam learning rate')
    ax.set_ylabel('Mean predicted $C_d$')
    ax.set_title('(b) Predicted drag vs. learning rate')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fpath = output_dir / 'lr_sweep_boxplot.pdf'
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    fig.savefig(fpath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved LR sweep plot -> {fpath}")

    # -- Also dump a small summary table --------------------------------------
    print("\n  LR sweep summary table:")
    print(f"  {'LR':>10s}  {'cost mean':>10s}  {'cost std':>10s}  {'Cd mean':>10s}  {'Cd std':>10s}")
    for lr in lr_vals:
        c = np.array(costs_by_lr[lr])
        d = np.array(cd_by_lr[lr])
        print(f"  {lr:>10.0e}  {c.mean():>10.5f}  {c.std():>10.6f}  "
              f"{d.mean():>10.5f}  {d.std():>10.6f}")


def _plot_multi_init_cost(results, step_indices, ax):
    """Cost scatter strip on a given Axes."""
    n_snaps = len(step_indices)
    for i, sidx in enumerate(step_indices):
        entries = [r for r in results if r['step_idx'] == sidx]
        ws   = [r for r in entries if r['seed'] == 'warm_start']
        rand = [r for r in entries if r['seed'] != 'warm_start']
        x_rand = np.full(len(rand), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(rand))
        ax.scatter(x_rand, [r['cost'] for r in rand],
                   c='steelblue', s=16, alpha=0.5, zorder=3, linewidth=0.0)
        if ws:
            ax.scatter(i, ws[0]['cost'], c='red', s=50, marker='D',
                       edgecolors='k', linewidths=0.5, zorder=2,
                       label='Warm start' if i == 0 else None)
    ax.set_xticks(range(n_snaps))
    ax.set_xticklabels([str(s) for s in step_indices], rotation=45, fontsize=7)
    ax.set_xlabel('MPC snapshot (step index)')
    ax.set_ylabel('Converged MPC cost')
    ax.set_title('Cost across random initializations')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)


def _plot_multi_init_action(results, step_indices, ax):
    """First-action scatter strip on a given Axes."""
    n_snaps = len(step_indices)
    for i, sidx in enumerate(step_indices):
        entries = [r for r in results if r['step_idx'] == sidx]
        ws   = [r for r in entries if r['seed'] == 'warm_start']
        rand = [r for r in entries if r['seed'] != 'warm_start']
        x_rand = np.full(len(rand), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(rand))
        ax.scatter(x_rand, [r['action_first'] for r in rand],
                   c='steelblue', s=18, alpha=0.6, zorder=3)
        if ws:
            ax.scatter(i, ws[0]['action_first'], c='red', s=50, marker='D',
                       edgecolors='k', linewidths=0.5, zorder=4,
                       label='Warm start' if i == 0 else None)
    ax.set_xticks(range(n_snaps))
    ax.set_xticklabels([str(s) for s in step_indices], rotation=45, fontsize=7)
    ax.set_xlabel('MPC snapshot (step index)')
    ax.set_ylabel('Optimal $a_0$')
    ax.set_title('First action across random initializations')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)


def plot_multi_init(results, output_dir):
    """Scatter / strip plot of converged cost across seeds per snapshot.

    Produces two separate files:
        multi_init_cost.pdf/png   -- converged MPC cost
        multi_init_action.pdf/png -- optimal first action a0
    """
    step_indices = sorted(set(r['step_idx'] for r in results))

    for plot_fn, stem in [
        (_plot_multi_init_cost,   'multi_init_cost'),
        (_plot_multi_init_action, 'multi_init_action'),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_fn(results, step_indices, ax)
        plt.tight_layout()
        fpath = output_dir / f'{stem}.pdf'
        fig.savefig(fpath, dpi=300, bbox_inches='tight')
        fig.savefig(fpath.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved multi-init plot -> {fpath}")

    # -- Per-snapshot summary -------------------------------------------------
    print("\n  Multi-init summary (random seeds only):")
    print(f"  {'step':>6s}  {'cost mean':>10s}  {'cost std':>10s}  "
          f"{'a0 mean':>10s}  {'a0 std':>10s}  {'cost range':>12s}")
    for sidx in step_indices:
        rand = [r for r in results if r['step_idx'] == sidx and r['seed'] != 'warm_start']
        c = np.array([r['cost'] for r in rand])
        a = np.array([r['action_first'] for r in rand])
        print(f"  {sidx:>6d}  {c.mean():>10.5f}  {c.std():>10.6f}  "
              f"{a.mean():>10.5f}  {a.std():>10.6f}  "
              f"[{c.min():.5f}, {c.max():.5f}]")


def plot_cost_traces(lr_results, output_dir):
    """
    Mean convergence trace per LR with min-max shaded band.
    Each snapshot's traces are normalized so that:
        cost_norm = (cost - cost_best) / (cost_initial - cost_best)
    where cost_initial = cost at iteration 0 of that snapshot (shared across LRs)
    and   cost_best    = min cost reached by any LR at that snapshot.
    This maps every trace to start near 1 and converge toward 0.
    """
    lr_vals = sorted(set(r['lr'] for r in lr_results))
    step_indices = sorted(set(r['step_idx'] for r in lr_results))

    # -- Collect raw traces per (snapshot, lr) --------------------------------
    # raw_traces[sidx][lr] = 1-D array of cost values
    raw_traces = {}
    for r in lr_results:
        if 'cost_trace' not in r:
            continue
        raw_traces.setdefault(r['step_idx'], {})[r['lr']] = np.array(r['cost_trace'])

    # -- Normalize per snapshot -----------------------------------------------
    # norm_traces_by_lr[lr] = list of normalized 1-D traces (one per snapshot)
    norm_traces_by_lr = {lr: [] for lr in lr_vals}

    for sidx in step_indices:
        if sidx not in raw_traces:
            continue
        snap_traces = raw_traces[sidx]  # dict: lr -> trace array
        if not snap_traces:
            continue

        # cost_initial: cost at iteration 0 (same initial guess -> same for all LRs)
        # Take the mean across LRs in case of small floating-point differences
        cost_initial = np.mean([t[0] for t in snap_traces.values()])

        # cost_best: lowest cost reached by any LR at any iteration
        cost_best = min(t.min() for t in snap_traces.values())

        denom = cost_initial - cost_best
        if abs(denom) < 1e-12:
            # All LRs stuck at the same cost -- skip this snapshot
            continue

        for lr in lr_vals:
            if lr in snap_traces:
                normed = (snap_traces[lr] - cost_best) / denom
                norm_traces_by_lr[lr].append(normed)

    # -- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(lr_vals)))

    for j, lr in enumerate(lr_vals):
        traces = norm_traces_by_lr[lr]
        if not traces:
            continue
        mat = np.array(traces)  # (n_snapshots, n_steps)
        iters = np.arange(mat.shape[1])
        mean = mat.mean(axis=0)
        lo   = mat.min(axis=0)
        hi   = mat.max(axis=0)

        ax.fill_between(iters, lo, hi, color=cmap[j], alpha=0.15)
        ax.plot(iters, mean, color=cmap[j], linewidth=1.5,
                label=f'lr={lr:.0e}')

    ax.set_xlabel('Adam iteration')
    ax.set_ylabel('Normalized cost (1 = initial, 0 = best)')
    ax.set_title('Optimizer convergence traces (normalized)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_ylim(bottom=-0.1)  # allow slight overshoot below 0 to be visible

    plt.tight_layout()
    fpath = output_dir / 'cost_convergence_traces.pdf'
    fig.savefig(fpath, dpi=300, bbox_inches='tight')
    fig.savefig(fpath.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved convergence traces -> {fpath}")


# ===============================================================================
#  Cache I/O
# ===============================================================================
def save_cache(cache_path, lr_results, init_results,
               lr_values, num_optim_steps, n_random_seeds,
               baseline_lr, source_h5):
    """Save all results (including cost traces) to HDF5 cache."""
    with h5py.File(cache_path, 'w') as f:
        # -- LR sweep ------------------------------------------------------
        if lr_results is not None:
            grp = f.create_group('lr_sweep')
            grp.create_dataset('step_idx',     data=[r['step_idx'] for r in lr_results])
            grp.create_dataset('lr',           data=[r['lr'] for r in lr_results])
            grp.create_dataset('cost',         data=[r['cost'] for r in lr_results])
            grp.create_dataset('mean_cd',      data=[r['mean_cd'] for r in lr_results])
            grp.create_dataset('action_first', data=[r['action_first'] for r in lr_results])
            # Cost traces: variable-length per entry but all same length in
            # practice (NUM_OPTIM_STEPS).  Store as 2-D array.
            traces = [r.get('cost_trace', []) for r in lr_results]
            if traces and len(traces[0]) > 0:
                grp.create_dataset('cost_traces', data=np.array(traces))

        # -- Multi-init ----------------------------------------------------
        if init_results is not None:
            grp = f.create_group('multi_init')
            grp.create_dataset('step_idx',     data=[r['step_idx'] for r in init_results])
            seeds_str = [str(r['seed']) for r in init_results]
            grp.create_dataset('seed', data=seeds_str)
            grp.create_dataset('cost',         data=[r['cost'] for r in init_results])
            grp.create_dataset('mean_cd',      data=[r['mean_cd'] for r in init_results])
            grp.create_dataset('action_first', data=[r['action_first'] for r in init_results])

        # -- Metadata ------------------------------------------------------
        f.attrs['lr_values']       = lr_values
        f.attrs['num_optim_steps'] = num_optim_steps
        f.attrs['n_random_seeds']  = n_random_seeds
        f.attrs['baseline_lr']     = baseline_lr
        f.attrs['source_h5']       = source_h5

    print(f"Cache saved -> {cache_path}")


def load_cache(cache_path):
    """Load results from HDF5 cache.  Returns (lr_results, init_results)."""
    lr_results = None
    init_results = None

    with h5py.File(cache_path, 'r') as f:
        # -- LR sweep ------------------------------------------------------
        if 'lr_sweep' in f:
            grp = f['lr_sweep']
            step_ids = grp['step_idx'][:]
            lrs      = grp['lr'][:]
            costs    = grp['cost'][:]
            cds      = grp['mean_cd'][:]
            a0s      = grp['action_first'][:]
            has_traces = 'cost_traces' in grp
            if has_traces:
                traces_arr = grp['cost_traces'][:]  # (N, n_steps)

            lr_results = []
            for i in range(len(step_ids)):
                entry = {
                    'step_idx':     int(step_ids[i]),
                    'lr':           float(lrs[i]),
                    'cost':         float(costs[i]),
                    'mean_cd':      float(cds[i]),
                    'action_first': float(a0s[i]),
                }
                if has_traces:
                    entry['cost_trace'] = traces_arr[i].tolist()
                lr_results.append(entry)

        # -- Multi-init ----------------------------------------------------
        if 'multi_init' in f:
            grp = f['multi_init']
            step_ids = grp['step_idx'][:]
            seeds    = [s.decode() if isinstance(s, bytes) else str(s)
                        for s in grp['seed'][:]]
            costs    = grp['cost'][:]
            cds      = grp['mean_cd'][:]
            a0s      = grp['action_first'][:]

            init_results = []
            for i in range(len(step_ids)):
                seed_val = seeds[i]
                # Restore original type: 'warm_start' stays str, others -> int
                if seed_val != 'warm_start':
                    try:
                        seed_val = int(seed_val)
                    except ValueError:
                        pass
                init_results.append({
                    'step_idx':     int(step_ids[i]),
                    'seed':         seed_val,
                    'cost':         float(costs[i]),
                    'mean_cd':      float(cds[i]),
                    'action_first': float(a0s[i]),
                })

    return lr_results, init_results


def _build_controller(original_model_id, checkpoints_base, results_base,
                      use_slim, slim_ckpt, slim_indices,
                      horizon, limits, lr, n_steps,
                      lam_rate, lam_effort, lam_smooth,
                      device):
    """Instantiate MPCControllerLatent (factored out to avoid duplication)."""
    print("\n== Initializing MPC controller ==")
    return MPCControllerLatent(
        model_identifier=original_model_id,
        checkpoints_base_dir=checkpoints_base,
        results_base_dir=results_base,
        horizon=horizon,
        limits=limits,
        device=device,
        lr_mpc=lr,
        num_optim_steps=n_steps,
        cost_lambda_rate=lam_rate,
        cost_lambda_effort=lam_effort,
        cost_lambda_smoothness=lam_smooth,
        use_slim_encoder=use_slim,
        slim_encoder_ckpt_path=slim_ckpt,
        selected_sensor_indices_path=slim_indices,
        actuator_delay=0,
    )


# ===============================================================================
#  Main
# ===============================================================================
if __name__ == "__main__":

    # +=======================================================================+
    # |  CONFIG -- edit these to match your setup                            |
    # +=======================================================================+
    cwd = Path.cwd()

    # -- Model / checkpoint configuration (same as MPC_onFOM.py) ----------
    original_model_id = '20250906_19_38_58'
    checkpoints_base  = cwd / "03_Checkpoints"
    results_base      = cwd / "04_Results"

    USE_SLIM_ENCODER = True
    case_name_for_sensor_optim_artifacts = "jet_2Dtruck_20250307_FMsignal_50000"
    slim_run_folder_name = ('20250906_19_38_58_LSTM_dim8_lb32_l1_h256_dr0.0_'
                            'lr0.001_bs512_wC_ntest0_SHAP_Optim_20250906_22_12_02')
    slim_folder_name = 'shap_optim_runs'

    # -- Path to a saved MPC_results.h5 from a previous MPC_onFOM run -----
    MPC_RESULTS_H5 = str(cwd / "gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20260328_213407/20250906_19_38_58_SlimMPC_20260328_213407_MPC_results.h5")  # <-- EDIT THIS

    # -- MPC baseline parameters (should match your production run) --------
    MPC_HORIZON       = 25
    CONTROL_LIMITS    = [-0.075, 0.075]
    NUM_OPTIM_STEPS   = 32*5       # higher than production (5) to ensure all LR converge. (production converges over 32 optimizations)
    COST_LAMBDA_RATE       = 0.0
    COST_LAMBDA_EFFORT     = 0.0
    COST_LAMBDA_SMOOTHNESS = 8.0

    CANCEL_DROPOUT = False

    # -- Experiment A: LR sweep -------------------------------------------
    LR_VALUES  = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    N_SNAPSHOTS_LR = 20    # number of evenly-spaced snapshots

    # -- Experiment B: Multi-initialization --------------------------------
    BASELINE_LR     = 1e-3
    N_RANDOM_SEEDS  = 20
    N_SNAPSHOTS_INIT = 15   # can be fewer -- each gets N_RANDOM_SEEDS + 1 runs
    MULTI_INIT_RNG_SEED = 42

    # -- Output -----------------------------------------------------------
    OUTPUT_DIR = cwd / "04_Results" / "lr_sensitivity"
    if CANCEL_DROPOUT:
        OUTPUT_DIR = OUTPUT_DIR / 'NoDropout'

    # +=======================================================================+
    # |  End of CONFIG                                                      |
    # +=======================================================================+

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -- Build slim encoder paths (same logic as MPC_onFOM.py) ------------
    SLIM_ENCODER_CKPT_PATH = None
    SELECTED_INDICES_PATH  = None
    if USE_SLIM_ENCODER:
        slim_base_ckp = (checkpoints_base / case_name_for_sensor_optim_artifacts
                         / slim_folder_name / slim_run_folder_name)
        slim_base_res = (results_base / case_name_for_sensor_optim_artifacts
                         / slim_folder_name / slim_run_folder_name)
        SLIM_ENCODER_CKPT_PATH = str(
            slim_base_ckp / f"{slim_run_folder_name}_phase2_slim_shap_encoder_slim_shap.pth.tar"
        )
        SELECTED_INDICES_PATH = str(
            slim_base_res / f"{slim_run_folder_name}_shap_selected_indices.txt"
        )

    # -- Cache file for expensive results ------------------------------------
    CACHE_H5 = OUTPUT_DIR / 'lr_sensitivity_results.h5'

    # ======================================================================
    #  Load from cache or compute
    # ======================================================================
    lr_results = None
    init_results = None

    if CACHE_H5.is_file():
        print(f"\n** Cache found: {CACHE_H5}")
        print("   Loading previous results -- delete the file to force recomputation.")
        lr_results, init_results = load_cache(CACHE_H5)
        if lr_results is not None:
            print(f"   LR sweep:    {len(lr_results)} entries loaded.")
        if init_results is not None:
            print(f"   Multi-init:  {len(init_results)} entries loaded.")

    # -- Experiment A -- LR sweep ------------------------------------------
    if lr_results is None:
        # -- Instantiate controller (only if we need to compute) -----------
        controller = _build_controller(
            original_model_id, checkpoints_base, results_base,
            USE_SLIM_ENCODER, SLIM_ENCODER_CKPT_PATH, SELECTED_INDICES_PATH,
            MPC_HORIZON, CONTROL_LIMITS, BASELINE_LR, NUM_OPTIM_STEPS,
            COST_LAMBDA_RATE, COST_LAMBDA_EFFORT, COST_LAMBDA_SMOOTHNESS,
            device,
        )
        all_snapshots = load_snapshots(MPC_RESULTS_H5,
                                       max(N_SNAPSHOTS_LR, N_SNAPSHOTS_INIT),
                                       controller.lookback)
        snapshots_lr = all_snapshots[:N_SNAPSHOTS_LR]

        print("\n== Experiment A: Learning-rate sweep ==")
        t0 = time.perf_counter()
        lr_results = run_lr_sweep(controller, snapshots_lr, LR_VALUES, NUM_OPTIM_STEPS, cancel_dropout=CANCEL_DROPOUT)
        print(f"LR sweep completed in {time.perf_counter() - t0:.1f} s")

    # -- Experiment B -- Multi-initialization ------------------------------
    if init_results is None:
        # Build controller & snapshots if not already done above
        if 'controller' not in dir():
            controller = _build_controller(
                original_model_id, checkpoints_base, results_base,
                USE_SLIM_ENCODER, SLIM_ENCODER_CKPT_PATH, SELECTED_INDICES_PATH,
                MPC_HORIZON, CONTROL_LIMITS, BASELINE_LR, NUM_OPTIM_STEPS,
                COST_LAMBDA_RATE, COST_LAMBDA_EFFORT, COST_LAMBDA_SMOOTHNESS,
                device,
            )
            all_snapshots = load_snapshots(MPC_RESULTS_H5,
                                           max(N_SNAPSHOTS_LR, N_SNAPSHOTS_INIT),
                                           controller.lookback)
        snapshots_init = all_snapshots[:N_SNAPSHOTS_INIT]

        print("\n== Experiment B: Multi-initialization convergence ==")
        t0 = time.perf_counter()
        init_results = run_multi_init(
            controller, snapshots_init, N_RANDOM_SEEDS, BASELINE_LR,
            NUM_OPTIM_STEPS, rng_seed=MULTI_INIT_RNG_SEED, cancel_dropout=CANCEL_DROPOUT,
        )
        print(f"Multi-init completed in {time.perf_counter() - t0:.1f} s")

    # ======================================================================
    #  Save cache (always, so partial runs are also cached)
    # ======================================================================
    save_cache(CACHE_H5, lr_results, init_results,
               LR_VALUES, NUM_OPTIM_STEPS, N_RANDOM_SEEDS,
               BASELINE_LR, MPC_RESULTS_H5)

    # ======================================================================
    #  Plot (always runs, even from cache -- fast)
    # ======================================================================
    print("\n== Generating plots ==")
    plot_lr_sweep(lr_results, OUTPUT_DIR)
    plot_cost_traces(lr_results, OUTPUT_DIR)
    plot_multi_init(init_results, OUTPUT_DIR)

    print(f"\nAll figures saved in -> {OUTPUT_DIR}")
    print("Done.")
