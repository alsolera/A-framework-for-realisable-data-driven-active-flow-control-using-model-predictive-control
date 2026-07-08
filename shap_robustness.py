# shap_robustness.py
#
# Robustness analysis of SHAP-based sensor importance scores.
# Runs GradientExplainer N times, each with a different random seed controlling
# the selection of background and explanation samples. Reports:
#   - Mean +/- std barplot of sensor SHAP importance (top-N sensors only).
#   - Top-k stability: grouped bar chart of stability scores for all k values,
#     showing only sensors that ever appear in any top-k.
#   - Spearman rank correlation matrix across runs.
#   - All results saved as .txt / .npz for downstream use.
#
# Only the background/explanation sample selection varies across runs.
# The model checkpoint (encoder weights) is fixed.
#
# Caching: each individual run is cached in 04_Results/SHAPstability/ under a
# filename that encodes the model ID, seed, and sampling parameters. Re-running
# the script skips any seed whose cache file already exists, so you can safely
# increase n_runs without repeating previous work.
#
# Usage: set CONFIG below and run.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from scipy.stats import spearmanr

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

from libs.models import TemporalEncoder
from libs.data import get_prepared_data
from libs.test_encoder_predictor import find_model_paths, load_checkpoint as load_model_checkpoint
from parameters import Args as OriginalArgs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RobustnessConfig:
    # ---- Model to analyse ----
    base_model_strdate_identifier: str = "20250906_19_38_58"  # <<< SET THIS

    # ---- Directories (must match your project layout) ----
    checkpoints_base_dir: str = "03_Checkpoints"
    results_base_dir: str = "04_Results"
    stability_dir: str = "04_Results/SHAPstability"  # per-run cache + final outputs

    # ---- SHAP sampling parameters (kept identical to original script) ----
    num_background_samples: int = 100
    num_explanation_samples: int = 500

    # ---- Robustness sweep ----
    n_runs: int = 10         # <<< number of independent seeds to test
    base_seed: int = 42      # seeds: base_seed, base_seed+1, ..., base_seed+n_runs-1

    # ---- Top-k stability analysis ----
    topk_values: List[int] = field(default_factory=lambda: [4])

    # ---- Plot options ----
    n_show_barplot: int = 20  # show only the top-N sensors in the mean/std barplot

    # ---- Hardware ----
    cuda: bool = True


CONFIG = RobustnessConfig()


# ---------------------------------------------------------------------------
# Shared plot style constants
# ---------------------------------------------------------------------------

_FS_LABEL = 8     # axis labels
_FS_TICK  = 7     # tick labels
_FS_ANNOT = 6     # in-cell annotations / footnotes
_LW_SPINE = 0.6   # spine / grid linewidth
_BAR_H    = 0.72  # bar height fraction
_COL_BAR  = "#4c72b0"  # default bar colour


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_mean_abs_shap(encoder: TemporalEncoder,
                            background: torch.Tensor,
                            explanation: torch.Tensor) -> np.ndarray:
    """
    Run GradientExplainer and aggregate to a single importance score per sensor.
    Mirrors the aggregation logic in rank_sensors_shap_and_train_slim.py exactly.
    """
    explainer = shap.GradientExplainer(encoder, background)

    # GradientExplainer requires train mode for cuDNN RNN layers
    encoder.train()
    shap_values_output = explainer.shap_values(explanation)
    encoder.eval()

    if isinstance(shap_values_output, list):
        # List of D arrays, each (N, L, F)
        stacked = np.array([np.abs(v) for v in shap_values_output])  # (D, N, L, F)
        summed_over_latents = np.sum(stacked, axis=0)                 # (N, L, F)
        mean_abs = np.mean(summed_over_latents, axis=(0, 1))          # (F,)

    elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 4:
        # Single array (N, L, F, D)
        abs_vals = np.abs(shap_values_output)
        summed_over_latents = np.sum(abs_vals, axis=3)                # (N, L, F)
        mean_abs = np.mean(summed_over_latents, axis=(0, 1))          # (F,)

    elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 3:
        # Single array (N, L, F)  scalar output
        abs_vals = np.abs(shap_values_output)
        mean_abs = np.mean(abs_vals, axis=(0, 1))                     # (F,)

    else:
        raise TypeError(
            f"Unexpected SHAP output type/shape: {type(shap_values_output)}, "
            f"ndim={getattr(shap_values_output, 'ndim', 'N/A')}"
        )

    return mean_abs


def _sample_shap_data(dataset, n_background: int, n_explanation: int,
                      rng: np.random.Generator, device: torch.device):
    """
    Draw background and explanation samples from *dataset* using *rng*
    so that each call with a different rng produces a different subset.
    """
    n_total = len(dataset)
    all_indices = rng.permutation(n_total)

    bg_indices = all_indices[:n_background]
    ex_indices = all_indices[n_background: n_background + n_explanation]

    # dataset[i] returns (s_t, ...)  grab only sensor tensor
    bg_tensors = torch.stack([dataset[int(i)][0] for i in bg_indices]).to(device)
    ex_tensors = torch.stack([dataset[int(i)][0] for i in ex_indices]).to(device)

    return bg_tensors, ex_tensors


def _run_cache_path(cache_dir: Path, model_id: str, seed: int,
                    n_bg: int, n_ex: int) -> Path:
    """
    Deterministic path for a single-run cache file.
    Encodes model ID, seed, and sampling parameters so that changing any of
    them produces a distinct filename and forces recomputation.
    """
    return cache_dir / f"{model_id}_seed{seed}_bg{n_bg}_ex{n_ex}_shap.npy"


def _load_run_cache(cache_path: Path):
    """Return cached array if the file exists, else None."""
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"  [cache hit]  loaded from {cache_path.name}")
        return data
    return None


def _save_run_cache(cache_path: Path, mean_abs: np.ndarray) -> None:
    np.save(cache_path, mean_abs)
    print(f"  [cache]  saved to {cache_path.name}")


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    for ext in (".png", ".pdf"):
        fig.savefig(out_dir / f"{stem}{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / stem}.*")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mean_std_barplot(mean_shap: np.ndarray, std_shap: np.ndarray,
                          topk_values: List[int], out_dir: Path,
                          filename_base: str) -> None:
    """
    Vertical barplot of the top-N sensors by mean SHAP +/- std, designed for
    a double-column layout (7 x 3 in).

    - X-axis: importance rank (1 = most important), labeled "Importance rank".
    - Y-axis: mean |SHAP| with std error bars.
    - Sensors within each top-k band are colored red; the rest blue.
    - Vertical dashed lines separate k bands; a small text label sits just
      above the top of the figure at the right edge of each band.
    - No legend (the k labels are self-explanatory).
    """
    n_sensors = len(mean_shap)
    n_show    = min(CONFIG.n_show_barplot, n_sensors)

    sort_idx  = np.argsort(mean_shap)[::-1]
    show_idx  = sort_idx[:n_show]
    show_mean = mean_shap[show_idx]
    show_std  = std_shap[show_idx]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    x_pos = np.arange(1, n_show + 1)   # ranks 1..n_show
    ax.bar(x_pos, show_mean, yerr=show_std,
           color=_COL_BAR, edgecolor="none",
           error_kw=dict(elinewidth=0.7, capsize=2.0, ecolor="#333333"),
           width=_BAR_H)

    # Vertical dashed lines between k bands, with small text annotation
    # placed just inside the top of the axes on the right side of each line.
    for k in topk_values:
        if k < n_show:
            ax.axvline(k + 0.5, color="grey", linestyle="--", linewidth=_LW_SPINE)
            ax.text(k + 0.55, 1.0, rf"$k={k}$",
                    transform=ax.get_xaxis_transform(),   # x in data, y in axes [0,1]
                    va="top", ha="left",
                    fontsize=_FS_ANNOT, color="grey")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in range(1, n_show + 1)], fontsize=_FS_TICK)
    ax.set_xlabel("Importance rank", fontsize=_FS_LABEL)
    ax.set_ylabel(r"Mean $|\mathrm{SHAP}|$ $\pm$ $\sigma$", fontsize=_FS_LABEL)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.tick_params(axis="x", length=0)

    ax.set_xlim(0.5, n_show + 0.5)
    ax.yaxis.set_tick_params(width=_LW_SPINE)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(_LW_SPINE)

    fig.tight_layout(pad=0.4)
    _save_fig(fig, out_dir, f"{filename_base}_mean_std_barplot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Device ----
    device = torch.device(
        "cuda" if torch.cuda.is_available() and CONFIG.cuda else "cpu"
    )
    print(f"Device: {device}")

    # ---- Locate checkpoint and load original args ----
    base_case_ckp_dir, base_case_results_dir = find_model_paths(
        CONFIG.base_model_strdate_identifier, CONFIG.checkpoints_base_dir
    )
    if base_case_ckp_dir is None or base_case_results_dir is None:
        print("ERROR: Could not locate model paths. Aborting.")
        return

    latent_file_pattern = f"{CONFIG.base_model_strdate_identifier}*latent_space.hdf5"
    latent_files = sorted(base_case_results_dir.glob(latent_file_pattern))
    base_latent_file = next(
        (f for f in latent_files if "_eval" not in f.name),
        latent_files[0] if latent_files else None
    )
    if base_latent_file is None:
        print("ERROR: No latent_space.hdf5 file found. Aborting.")
        return

    with h5py.File(base_latent_file, "r") as f:
        original_args_dict = json.loads(f.attrs["args"])
    original_args = OriginalArgs(**original_args_dict)

    # ---- Output / cache directory ----
    # All per-run caches and final outputs live under 04_Results/SHAPstability/.
    # The subdirectory encodes sampling parameters so that changing them
    # creates a fresh folder and avoids mixing incompatible cached runs.
    out_dir = (Path(CONFIG.stability_dir) /
               f"{CONFIG.base_model_strdate_identifier}"
               f"_bg{CONFIG.num_background_samples}"
               f"_ex{CONFIG.num_explanation_samples}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename base for aggregated outputs (no timestamp -- stable across reruns)
    filename_base = (f"{CONFIG.base_model_strdate_identifier}"
                     f"_bg{CONFIG.num_background_samples}"
                     f"_ex{CONFIG.num_explanation_samples}"
                     f"_runs{CONFIG.n_runs}"
                     f"_seeds{CONFIG.base_seed}-{CONFIG.base_seed + CONFIG.n_runs - 1}")
    print(f"Output / cache directory: {out_dir}")

    # ---- Build dataloader ----
    loader_args = OriginalArgs(**original_args_dict)
    loader_args.batch_size = CONFIG.num_background_samples + CONFIG.num_explanation_samples
    loader_args.n_test = 0
    loader_args.DATA_TO_GPU = False
    dataloader, _ = get_prepared_data(loader_args, device,
                                      shuffle_train=False, shuffle_test=False)
    dataset = dataloader.dataset

    # ---- Determine input dimension ----
    sample_s_t = dataset[0][0]
    in_dim = sample_s_t.shape[-1]
    print(f"Number of sensors (input dim): {in_dim}")

    # ---- Load encoder (fixed weights for all runs) ----
    encoder = TemporalEncoder(
        input_dim=in_dim,
        latent_dim=original_args.latent_dim,
        hidden_dim=original_args.enc_hidden_dim,
        num_layers=original_args.n_layers,
        dropout_rate=original_args.dropout,
    ).to(device)
    enc_path = base_case_ckp_dir / f"{original_args.modelname}_encoder.pth.tar"
    load_model_checkpoint(encoder, str(enc_path), device)
    encoder.eval()
    print(f"Loaded encoder from {enc_path}")

    # ---- Robustness loop (with per-run caching) ----
    all_shap   = np.zeros((CONFIG.n_runs, in_dim))
    n_computed = 0
    n_cached   = 0

    for run_idx in range(CONFIG.n_runs):
        seed = CONFIG.base_seed + run_idx
        print(f"\n[Run {run_idx + 1}/{CONFIG.n_runs}]  seed={seed}")

        cache_path = _run_cache_path(
            out_dir, CONFIG.base_model_strdate_identifier, seed,
            CONFIG.num_background_samples, CONFIG.num_explanation_samples
        )

        mean_abs = _load_run_cache(cache_path)

        if mean_abs is None:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)

            background, explanation = _sample_shap_data(
                dataset,
                CONFIG.num_background_samples,
                CONFIG.num_explanation_samples,
                rng,
                device,
            )
            print(f"  Background: {background.shape}  |  Explanation: {explanation.shape}")

            mean_abs = _compute_mean_abs_shap(encoder, background, explanation)
            _save_run_cache(cache_path, mean_abs)
            n_computed += 1
        else:
            n_cached += 1

        all_shap[run_idx] = mean_abs
        print(f"  Max={mean_abs.max():.5f}  Min={mean_abs.min():.5f}")

    print(f"\nRuns completed: {n_computed} computed, {n_cached} loaded from cache.")

    # ---- Summary statistics ----
    mean_shap = all_shap.mean(axis=0)
    std_shap  = all_shap.std(axis=0)
    cv_shap   = np.where(mean_shap > 0, std_shap / mean_shap, 0.0)

    rhos = []
    for i in range(CONFIG.n_runs):
        for j in range(i + 1, CONFIG.n_runs):
            r, _ = spearmanr(all_shap[i], all_shap[j])
            rhos.append(r)
    mean_rho = np.mean(rhos)
    print(f"\nMean pairwise Spearman rho: {mean_rho:.4f}")

    print("\nTop-k consensus (sensors present in ALL runs):")
    for k in CONFIG.topk_values:
        topk_sets = [set(np.argsort(all_shap[r])[::-1][:k])
                     for r in range(CONFIG.n_runs)]
        consensus = set.intersection(*topk_sets)
        pct = len(consensus) / k * 100
        print(f"  k={k:2d}:  consensus sensors = {sorted(consensus)}"
              f"  ({len(consensus)}/{k} = {pct:.0f}%)")

    # ---- Save raw data ----
    npz_path = out_dir / f"{filename_base}_all_shap_runs.npz"
    np.savez(
        npz_path,
        all_shap=all_shap,
        mean_shap=mean_shap,
        std_shap=std_shap,
        cv_shap=cv_shap,
        seeds=np.arange(CONFIG.base_seed, CONFIG.base_seed + CONFIG.n_runs),
    )
    print(f"\nSaved raw SHAP arrays to {npz_path}")

    # Human-readable summary table
    txt_path = out_dir / f"{filename_base}_summary.txt"
    sort_idx = np.argsort(mean_shap)[::-1]
    with open(txt_path, "w") as fh:
        fh.write("SHAP robustness summary\n")
        fh.write(f"Model: {CONFIG.base_model_strdate_identifier}\n")
        fh.write(f"n_runs={CONFIG.n_runs}, "
                 f"n_background={CONFIG.num_background_samples}, "
                 f"n_explanation={CONFIG.num_explanation_samples}\n")
        fh.write(f"Mean pairwise Spearman rho: {mean_rho:.4f}\n\n")
        fh.write(f"{'Rank':>4}  {'Sensor':>6}  {'Mean SHAP':>10}  "
                 f"{'Std SHAP':>10}  {'CV':>8}\n")
        fh.write("-" * 46 + "\n")
        for rank, sid in enumerate(sort_idx, start=1):
            fh.write(f"{rank:>4}  {sid:>6}  {mean_shap[sid]:>10.6f}  "
                     f"{std_shap[sid]:>10.6f}  {cv_shap[sid]:>8.4f}\n")
    print(f"Saved summary table to {txt_path}")

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_mean_std_barplot(mean_shap, std_shap, CONFIG.topk_values,
                          out_dir, filename_base)

    print("\nDone.")


if __name__ == "__main__":
    main()