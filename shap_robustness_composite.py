# shap_robustness_composite.py
#
# Composite 3-panel figure for the SHAP-based sensor selection robustness analysis:
#   (a) SHAP importance stability over N seeds (mean +/- sigma, top-K ranked sensors)
#   (b) Local Pearson correlation matrix around the four SHAP-selected sensors,
#       grouped as left side, rear face, right side.
#   (c) Leave-one-out (LOO) retraining: relative MAE for Cd and Cl when each of the
#       4 selected sensors is removed (3-sensor slim encoders), with the 4-sensor
#       baseline shown as a horizontal reference. Bars are labelled by physical
#       position rather than sensor index.
#
# Caches reused from previous runs:
#   - shap_robustness.py:               04_Results/SHAPstability/.../*.npz
#   - rank_sensors_shap_and_train_slim: *_shap_selected_indices.txt
#   - evaluate_sensor_subsets.py:       subset_evaluation/slim_encoder_4sensors.pth.tar
#   - this script:                      loo_evaluation/loo_results_seed<seed>.json
#
# Usage:  edit CFG below, then run.

import json
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import List

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

from libs.models import TemporalEncoder, ForceDecoder
from libs.data import get_prepared_data, loadData
from libs.test_encoder_predictor import find_model_paths, load_checkpoint
from parameters import Args as OriginalArgs
from evaluate_sensor_subsets import (
    train_slim_encoder,
    evaluate_force_error,
    load_raw_forces,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompositeConfig:
    # Model and base SHAP run
    base_model_strdate_identifier: str = "20250906_19_38_58"
    checkpoints_base_dir: str = "03_Checkpoints"
    results_base_dir: str = "04_Results"
    stability_dir: str = "04_Results/SHAPstability"

    # Sampling parameters used in shap_robustness.py
    num_background_samples: int = 100
    num_explanation_samples: int = 500
    n_runs: int = 10
    base_seed: int = 42

    # Selection
    top_n_sensors_to_select: int = 4
    enforce_symmetry: bool = False

    # LOO retraining
    loo_seeds: List[int] = field(default_factory=lambda: [4321])
    loo_epochs: int = 100
    loo_lr: float = 1e-3
    loo_batch_size: int = 256
    n_test_for_eval: int = 5000

    # Stability panel
    n_show_barplot: int = 10

    # Correlation panel
    trailing_window: int = 5   # number of sensors before each trailing edge

    # Hardware
    cuda: bool = True


CFG = CompositeConfig()


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_FS_LABEL = 9
_FS_TICK = 8
_FS_ANNOT = 7
_LW_SPINE = 0.6
_COL_BAR = "tab:blue"
_COL_CD = "tab:blue"
_COL_CL = "tab:orange"
_COL_SEL = "tab:green"


# Physical surface conventions (encoded once in the script)
#   indices 0-39   : top surface,    sensors ordered front -> rear (39 = top-rear corner)
#   indices 40-79  : bottom surface, sensors ordered front -> rear (79 = bottom-rear corner)
#   indices 80-89  : rear face,      sensors ordered top   -> bottom
TOP_RANGE    = (0, 39)
BOTTOM_RANGE = (40, 79)
REAR_RANGE   = (80, 89)

# Descriptive labels for the four corners (selected by SHAP: 39, 79, 80, 89)
CORNER_LABEL = {
    39: "Top trailing edge",
    79: "Bottom trailing edge",
    80: "Rear top",
    89: "Rear bottom",
}
# Short labels for the x-axis of the LOO panel.
# These mirror the directional labels used in panel (b):
#   "Left"  = bottom side of the truck (front-to-rear)
#   "Right" = top side
#   "Rear"  = rear face (top-to-bottom)
# The four selected sensors are the corners of this trio of surfaces.
CORNER_LABEL_SHORT = {
    39: "Right",
    79: "Left",
    80: "Rear R",
    89: "Rear L",
}


# ---------------------------------------------------------------------------
# Helpers: cached SHAP outputs
# ---------------------------------------------------------------------------

def _stability_out_dir(cfg: CompositeConfig) -> Path:
    return (Path(cfg.stability_dir) /
            f"{cfg.base_model_strdate_identifier}"
            f"_bg{cfg.num_background_samples}"
            f"_ex{cfg.num_explanation_samples}")


def _stability_npz_path(cfg: CompositeConfig) -> Path:
    stem = (f"{cfg.base_model_strdate_identifier}"
            f"_bg{cfg.num_background_samples}"
            f"_ex{cfg.num_explanation_samples}"
            f"_runs{cfg.n_runs}"
            f"_seeds{cfg.base_seed}-{cfg.base_seed + cfg.n_runs - 1}")
    return _stability_out_dir(cfg) / f"{stem}_all_shap_runs.npz"


def _shap_run_artifact_dir(cfg: CompositeConfig, case_name: str) -> Path:
    return (Path(cfg.results_base_dir) / case_name /
            "sensor_selection" / cfg.base_model_strdate_identifier)


def _load_selected_indices(cfg: CompositeConfig, case_name: str) -> np.ndarray:
    artifact_dir = _shap_run_artifact_dir(cfg, case_name)
    suffix = "sym" if cfg.enforce_symmetry else "asym"
    run_name = (f"{cfg.base_model_strdate_identifier}_"
                f"{cfg.top_n_sensors_to_select}sensors_{suffix}")
    candidates = list(artifact_dir.glob(f"{run_name}_shap_selected_indices.txt"))
    if not candidates:
        candidates = list(artifact_dir.glob("*shap_selected_indices.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find SHAP selected-indices file in {artifact_dir}.")
    path = candidates[0]
    print(f"Loading SHAP-selected indices from: {path}")
    return np.loadtxt(path, dtype=int)


# ---------------------------------------------------------------------------
# Panel (a): SHAP stability
# ---------------------------------------------------------------------------

def load_stability_data(cfg: CompositeConfig):
    npz_path = _stability_npz_path(cfg)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Stability npz not found at {npz_path}. Run shap_robustness.py first.")
    data = np.load(npz_path)
    return data["mean_shap"], data["std_shap"]


def plot_stability(ax, mean_shap, std_shap, k_value, n_show, cfg: CompositeConfig):
    n_sensors = len(mean_shap)
    n_show = min(n_show, n_sensors)

    sort_idx = np.argsort(mean_shap)[::-1]
    show_mean = mean_shap[sort_idx[:n_show]]
    show_std = std_shap[sort_idx[:n_show]]

    x_pos = np.arange(1, n_show + 1)
    ax.bar(x_pos, show_mean, yerr=show_std,
           color=_COL_BAR, edgecolor="none",
           error_kw=dict(elinewidth=0.7, capsize=2.0, ecolor="#333333"),
           width=0.72)

    if k_value < n_show:
        ax.axvline(k_value + 0.5, color="grey", linestyle="--", linewidth=_LW_SPINE)
        ax.text(k_value + 0.6, 0.95, rf"$k={k_value}$",
                transform=ax.get_xaxis_transform(),
                va="top", ha="left",
                fontsize=_FS_ANNOT, color="grey")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in range(1, n_show + 1)], fontsize=_FS_TICK)
    ax.set_xlabel("Importance rank", fontsize=_FS_LABEL)
    ax.set_ylabel(r"Mean $|\mathrm{SHAP}|$ $\pm$ $\sigma$", fontsize=_FS_LABEL)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(0.5, n_show + 0.5)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(_LW_SPINE)


# ---------------------------------------------------------------------------
# Panel (b): local correlation matrix
# ---------------------------------------------------------------------------

def compute_full_correlation(original_args: OriginalArgs) -> np.ndarray:
    print("Loading training data to compute sensor correlation...")
    class _Args:
        datafile = original_args.datafile
        augment_with_symmetry = False
    (p_unscaled, _, _, _, _, _, _, _, _, _, _) = loadData(
        _Args, sel_coefs=['Cd', 'Cl'], printer=False)
    print(f"  Sensor data shape: {p_unscaled.shape}")
    return np.corrcoef(p_unscaled.T)


def _build_correlation_subset(cfg: CompositeConfig):
    """
    Build the ordered list of sensor indices to display in the correlation
    panel, and the group spans (start_pos, end_pos, label) for the axes.

    Order along the diagonal, from one trailing edge around the rear to the other:

        Left side (top trailing edge window):
            [39-w, ..., 39]        -- last `w+1` sensors of the top surface
            length = w + 1

        Rear face (full):
            [80, 81, ..., 89]      -- all 10 rear-face sensors
            length = 10

        Right side (bottom trailing edge window), reversed so adjacent sensors
        are spatially adjacent across the corner 89 -> 79:
            [79, 78, ..., 79-w]
            length = w + 1

    With w = 5 -> total = 6 + 10 + 6 = 22 sensors.
    """
    w = cfg.trailing_window

    left  = list(range(TOP_RANGE[1]    - w, TOP_RANGE[1]    + 1))   # 34..39
    rear  = list(range(REAR_RANGE[0],         REAR_RANGE[1]   + 1)) # 80..89
    right = list(range(BOTTOM_RANGE[1],       BOTTOM_RANGE[1] - w - 1, -1))  # 79..74

    ordered = np.array(left + rear + right, dtype=int)

    groups = [
        (0,                len(left) - 1,                       "Left"),
        (len(left),        len(left) + len(rear) - 1,           "Rear"),
        (len(left) + len(rear), len(ordered) - 1,               "Right"),
    ]
    return ordered, groups


def plot_correlation(ax, corr_full, selected_indices, cfg: CompositeConfig):
    ordered, groups = _build_correlation_subset(cfg)

    # Sub-matrix in the chosen ordering
    sub = corr_full[np.ix_(ordered, ordered)]

    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   origin="upper", aspect="equal")

    n = len(ordered)

    # --- Group dividers and labels ---
    for (start, end, _label) in groups[:-1]:
        ax.axhline(end + 0.5, color="black", linewidth=0.7, alpha=0.6)
        ax.axvline(end + 0.5, color="black", linewidth=0.7, alpha=0.6)

    # Group tick labels at block centres
    centres = [(s + e) / 2 for (s, e, _l) in groups]
    labels  = [l for (_s, _e, l) in groups]
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=_FS_TICK)
    ax.set_yticks(centres)
    ax.set_yticklabels(labels, fontsize=_FS_TICK, rotation=90, va="center")
    ax.tick_params(axis="both", length=0, pad=2)

    # --- Mark selected sensors with red margin markers ---
    selected_set = set(selected_indices.tolist())
    sel_positions = [i for i, idx in enumerate(ordered) if idx in selected_set]

    # Tiny red triangles just outside the axes pointing in
    for pos in sel_positions:
        # top edge marker (pointing down)
        ax.plot(pos, -0.5, marker="v", color=_COL_SEL,
                markersize=4.5, clip_on=False, zorder=10)
        # left edge marker (pointing right)
        ax.plot(-0.5, pos, marker=">", color=_COL_SEL,
                markersize=4.5, clip_on=False, zorder=10)

    # Faint highlight box around each selected sensor on the diagonal
    for pos in sel_positions:
        ax.add_patch(plt.Rectangle((pos - 0.5, pos - 0.5), 1, 1,
                                    fill=False, edgecolor=_COL_SEL,
                                    linewidth=0.8, zorder=8))

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # origin='upper'

    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_linewidth(_LW_SPINE)

    return im


# ---------------------------------------------------------------------------
# Panel (c): LOO experiment
# ---------------------------------------------------------------------------

def run_loo_experiment(cfg: CompositeConfig,
                       original_args: OriginalArgs,
                       selected_indices: np.ndarray,
                       device: torch.device) -> dict:
    base_case_ckp_dir, base_case_results_dir = find_model_paths(
        cfg.base_model_strdate_identifier, cfg.checkpoints_base_dir)
    if base_case_ckp_dir is None:
        raise FileNotFoundError("Base model checkpoint dir not found.")

    base_artifact_dir = (Path(cfg.results_base_dir) / original_args.case /
                         "sensor_selection" / cfg.base_model_strdate_identifier)
    loo_dir = base_artifact_dir / "loo_evaluation"
    loo_dir.mkdir(parents=True, exist_ok=True)
    cache_file = loo_dir / f"loo_results_seed{cfg.loo_seeds[0]}.json"

    if cache_file.exists():
        print(f"Loading cached LOO results from {cache_file}")
        with open(cache_file, "r") as fh:
            return json.load(fh)

    # --- Scaling tensors ---
    latent_file_pattern = f"{cfg.base_model_strdate_identifier}*latent_space.hdf5"
    base_latent_file = next(
        (f for f in sorted(base_case_results_dir.glob(latent_file_pattern))
         if "_eval" not in f.name), None)
    with h5py.File(base_latent_file, "r") as f:
        forces_mean = f["forces_mean"][:]
        forces_std = f["forces_std"][:]
    forces_mean_t = torch.tensor(forces_mean, dtype=torch.float32, device=device)
    forces_std_t = torch.tensor(forces_std, dtype=torch.float32, device=device)

    # --- Dataloaders ---
    loader_args_train = deepcopy(original_args)
    loader_args_train.batch_size = cfg.loo_batch_size
    loader_args_train.n_test = cfg.n_test_for_eval
    loader_args_train.DATA_TO_GPU = False
    dl_train_slim, dl_test_slim = get_prepared_data(
        loader_args_train, device, shuffle_train=True, shuffle_test=False)

    sample_s_t = dl_train_slim.dataset[0][0]
    in_dim_orig = sample_s_t.shape[-1]

    encoder_orig = TemporalEncoder(
        input_dim=in_dim_orig, latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim, num_layers=original_args.n_layers
    ).to(device)
    force_decoder = ForceDecoder(
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.forces_hidden_dim,
        arch=original_args.force_decoder_arch
    ).to(device)
    enc_path = base_case_ckp_dir / f"{original_args.modelname}_encoder.pth.tar"
    force_path = base_case_ckp_dir / f"{original_args.modelname}_force.pth.tar"
    load_checkpoint(encoder_orig, str(enc_path), device)
    load_checkpoint(force_decoder, str(force_path), device)
    encoder_orig.eval()
    force_decoder.eval()

    loader_args_eval = deepcopy(original_args)
    project_root = Path(cfg.checkpoints_base_dir).parent
    eval_paths = original_args.eval_dataset
    if isinstance(eval_paths, str):
        eval_paths = [eval_paths]
    loader_args_eval.datafile = [str(project_root / p) for p in eval_paths]
    loader_args_eval.batch_size = 512
    loader_args_eval.n_test = 0
    loader_args_eval.augment_with_symmetry = False
    dl_force_eval, _ = get_prepared_data(
        loader_args_eval, device, shuffle_train=False, shuffle_test=False)
    raw_forces_eval = load_raw_forces(loader_args_eval)
    forces_std_test = np.std(raw_forces_eval, axis=0)

    class _SlimCfg:
        slim_training_epochs = cfg.loo_epochs
        slim_training_lr = cfg.loo_lr

    selected_sorted = np.sort(selected_indices)

    # --- 4-sensor baseline (reuse if cached by evaluate_sensor_subsets.py) ---
    baseline_ckpt = (base_artifact_dir / "subset_evaluation"
                     / f"slim_encoder_{len(selected_sorted)}sensors.pth.tar")
    if not baseline_ckpt.exists():
        print("Baseline 4-sensor slim encoder not cached; training it now...")
        slim_baseline = TemporalEncoder(
            input_dim=len(selected_sorted), latent_dim=original_args.latent_dim,
            hidden_dim=original_args.enc_hidden_dim,
            num_layers=original_args.n_layers
        ).to(device)
        baseline_ckpt.parent.mkdir(parents=True, exist_ok=True)
        train_slim_encoder(
            slim_baseline, encoder_orig, dl_train_slim, dl_test_slim,
            selected_sorted, _SlimCfg, device, baseline_ckpt
        )
    slim_baseline = TemporalEncoder(
        input_dim=len(selected_sorted), latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim,
        num_layers=original_args.n_layers
    ).to(device)
    load_checkpoint(slim_baseline, str(baseline_ckpt), device)
    mae_cd_b, mae_cl_b, _, _ = evaluate_force_error(
        slim_baseline, force_decoder, dl_force_eval, raw_forces_eval,
        selected_sorted, device, forces_mean_t, forces_std_t)
    baseline = {
        "norm_mae_cd": float(mae_cd_b / forces_std_test[0]),
        "norm_mae_cl": float(mae_cl_b / forces_std_test[1]),
    }
    print(f"Baseline (4 sensors): rel MAE Cd={baseline['norm_mae_cd']:.4f}, "
          f"Cl={baseline['norm_mae_cl']:.4f}")

    # --- LOO ---
    seed = cfg.loo_seeds[0]
    np.random.seed(seed)
    torch.manual_seed(seed)

    loo_results = []
    for removed in selected_sorted:
        kept = np.array([i for i in selected_sorted if i != removed])
        ckpt = loo_dir / f"slim_loo_remove{removed}_seed{seed}.pth.tar"

        slim = TemporalEncoder(
            input_dim=len(kept), latent_dim=original_args.latent_dim,
            hidden_dim=original_args.enc_hidden_dim,
            num_layers=original_args.n_layers
        ).to(device)

        if not ckpt.exists():
            print(f"\n[LOO] Training slim encoder without sensor {removed} "
                  f"(seed {seed})...")
            train_slim_encoder(
                slim, encoder_orig, dl_train_slim, dl_test_slim,
                kept, _SlimCfg, device, ckpt
            )
        else:
            print(f"[LOO] Found cached ckpt for removed={removed}, seed={seed}")

        load_checkpoint(slim, str(ckpt), device)
        mae_cd, mae_cl, _, _ = evaluate_force_error(
            slim, force_decoder, dl_force_eval, raw_forces_eval,
            kept, device, forces_mean_t, forces_std_t)
        entry = {
            "removed_sensor": int(removed),
            "kept_sensors": kept.tolist(),
            "norm_mae_cd": float(mae_cd / forces_std_test[0]),
            "norm_mae_cl": float(mae_cl / forces_std_test[1]),
        }
        loo_results.append(entry)
        print(f"  removed {removed}: rel MAE Cd={entry['norm_mae_cd']:.4f}, "
              f"Cl={entry['norm_mae_cl']:.4f}")

    results = {"baseline": baseline, "loo": loo_results}
    with open(cache_file, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved LOO results to {cache_file}")
    return results


def plot_loo(ax, loo_results: dict):
    loo = loo_results["loo"]
    baseline = loo_results["baseline"]

    # Order bars to follow the spatial order used in panel (b):
    #   Top trailing edge (39), Rear top (80), Rear bottom (89), Bottom trailing edge (79)
    order = [39, 80, 89, 79]
    by_id = {e["removed_sensor"]: e for e in loo}

    cd_vals = [by_id[i]["norm_mae_cd"] for i in order]
    cl_vals = [by_id[i]["norm_mae_cl"] for i in order]
    labels  = [CORNER_LABEL_SHORT[i] for i in order]

    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, cd_vals, width=w, color=_COL_CD,
           edgecolor="none", label=r"$C_d$")
    ax.bar(x + w / 2, cl_vals, width=w, color=_COL_CL,
           edgecolor="none", label=r"$C_l$")

    ax.axhline(baseline["norm_mae_cd"], color=_COL_CD,
               linestyle=":", linewidth=1.2,
               label=r"$C_d$ 4-sensor baseline")
    ax.axhline(baseline["norm_mae_cl"], color=_COL_CL,
               linestyle=":", linewidth=1.2,
               label=r"$C_l$ 4-sensor baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_FS_TICK)
    ax.set_xlabel("Removed sensor", fontsize=_FS_LABEL)
    ax.set_ylabel(r"Relative error ($L_1/\sigma$)", fontsize=_FS_LABEL)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    # Legend above the plot area in two columns; the figure-level layout leaves
    # space for it above ax_c.
    ax.legend(fontsize=_FS_ANNOT, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(bottom=0)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(_LW_SPINE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() and CFG.cuda
                          else "cpu")
    print(f"Device: {device}")

    base_case_ckp_dir, base_case_results_dir = find_model_paths(
        CFG.base_model_strdate_identifier, CFG.checkpoints_base_dir)
    latent_file_pattern = f"{CFG.base_model_strdate_identifier}*latent_space.hdf5"
    base_latent_file = next(
        (f for f in sorted(base_case_results_dir.glob(latent_file_pattern))
         if "_eval" not in f.name), None)
    with h5py.File(base_latent_file, "r") as f:
        original_args_dict = json.loads(f.attrs["args"])
    original_args = OriginalArgs(**original_args_dict)

    out_dir = (Path(CFG.results_base_dir) / original_args.case /
               "sensor_selection" / CFG.base_model_strdate_identifier
               / "composite_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cached or recomputed inputs
    mean_shap, std_shap = load_stability_data(CFG)
    corr_full = compute_full_correlation(original_args)
    selected_indices = _load_selected_indices(CFG, original_args.case)
    print(f"Selected sensors: {selected_indices.tolist()}")
    loo_results = run_loo_experiment(CFG, original_args, selected_indices, device)

    # --- Composite figure ---
    # 7-inch wide for a journal double-column figure.
    fig = plt.figure(figsize=(7.0, 2.6))

    # Give panel (b) a generous width so the correlation matrix (aspect='equal')
    # can grow as large as the panel height allows.
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.00, 1.5, 1.10],
        wspace=0.1,
        left=0.02, right=0.98,
        bottom=0.10, top=0.82,   # leaves room above (c) for its legend
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_stability(ax_a, mean_shap, std_shap,
                   k_value=CFG.top_n_sensors_to_select,
                   n_show=CFG.n_show_barplot, cfg=CFG)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    im = plot_correlation(ax_b, corr_full, selected_indices, CFG)
    cax = inset_axes(
        ax_b,
        width="100%", height="4%",
        loc="lower center",
        bbox_to_anchor=(0., 1.05, 1, 1),
        bbox_transform=ax_b.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Pearson $r$", fontsize=_FS_LABEL)
    cbar.ax.tick_params(labelsize=_FS_TICK)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.outline.set_visible(False)
    cbar.solids.set_rasterized(False)

    plot_loo(ax_c, loo_results)

    # Vertically-aligned panel labels via figure coordinates.
    # All three sit at the same y just above the top edge of the axes.
    label_y = 0.93
    for ax_obj, tag in [(ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")]:
        bbox = ax_obj.get_position()
        fig.text(bbox.x0 - 0.05, label_y, tag,
                 fontsize=_FS_LABEL + 1, va="bottom", ha="left")

    out_base = out_dir / "shap_robustness_composite"
    for ext in (".png", ".pdf"):
        fig.savefig(out_base.with_suffix(ext), bbox_inches="tight")
        print(f"Saved {out_base.with_suffix(ext)}")
    plt.close(fig)


if __name__ == "__main__":
    main()