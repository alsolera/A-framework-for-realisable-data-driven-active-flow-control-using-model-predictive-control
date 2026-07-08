#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Utilities: run discovery
# ---------------------------

def get_latest_run_folder(base_dir: str) -> str | None:
    """
    Find subfolder with the newest YYYYMMDD_HHMMSS pattern.
    """
    if not os.path.isdir(base_dir):
        print(f"[ERR] base_dir does not exist: {base_dir}")
        return None

    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    ts_pat = re.compile(r'(\d{8}_\d{6})')
    latest_dt = None
    latest_name = None
    for d in all_dirs:
        m = ts_pat.search(d)
        if not m:
            continue
        try:
            dt = datetime.datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            continue
        if latest_dt is None or dt > latest_dt:
            latest_dt = dt
            latest_name = d
    if latest_name is None:
        return None
    return os.path.join(base_dir, latest_name)


def find_largest_matching(directory: Path, base_filename: str) -> Path:
    """
    Pick the largest file whose name starts with `base_filename` in `directory` (recursively).
    Useful for OpenFOAM postProcessing where files may be split/rotated.
    """
    files = list(directory.rglob(f"{base_filename}*"))
    if not files:
        raise FileNotFoundError(f"No files like '{base_filename}*' under {directory}")
    return max(files, key=lambda p: p.stat().st_size)


# ---------------------------
# Robust header parsing
# ---------------------------

def detect_force_coeff_columns(coeff_path: Path) -> tuple[int, int]:
    """
    Read the header of coefficient.dat and detect 0-based indices for Cd and Cl
    (excluding the 'Time' column). Returns (cd_idx, cl_idx).
    Raises if we cannot find them.
    """
    header_lines = []
    with open(coeff_path, "r") as f:
        for _ in range(100):
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.lstrip().startswith("#"):
                header_lines.append(line.strip("# \n\t"))
            else:
                # Rewind to first data line
                f.seek(pos)
                break

    # Look for a header row listing columns after "Time"
    tokens = None
    for line in header_lines:
        lo = line.lower()
        if "time" in lo and ("cd" in lo or "cl" in lo):
            # split on whitespace/punctuations
            tks = re.split(r"[\s,;:|()\[\]\t]+", lo)
            tks = [t for t in tks if t]
            tokens = tks
            break

    if not tokens:
        raise RuntimeError(f"Could not detect Cd/Cl columns from header of {coeff_path}")

    # Drop only the *first* 'time'
    if "time" in tokens:
        tpos = tokens.index("time")
        tokens = tokens[tpos+1:]

    # Prefer exact 'cd' and 'cl'; allow common variants for Cl
    cd_idx = tokens.index("cd") if "cd" in tokens else None
    cl_idx = None
    for key in ("cl", "cly", "cl_y", "cl(y)", "cltot", "cl_total"):
        if key in tokens:
            cl_idx = tokens.index(key)
            break

    if cd_idx is None or cl_idx is None:
        raise RuntimeError(f"Header parsed but Cd/Cl not found. Tokens={tokens}")

    # Return 0-based indices counting *after* Time
    return cd_idx, cl_idx


# ---------------------------
# Incremental file monitors
# ---------------------------

class DataMonitor:
    """
    Incremental reader for an appending ascii file.
    columns_to_read: 0-based indices in the split line.
    """
    def __init__(self, file_path: Path, columns_to_read: list[int]):
        self.file_path = Path(file_path)
        self.columns_to_read = list(columns_to_read)
        self._pos = 0
        self.data = [[] for _ in columns_to_read]
        self._last_times = []

    def read_new(self) -> bool:
        try:
            with open(self.file_path, "r") as f:
                f.seek(self._pos)
                new = f.read()
                self._pos = f.tell()
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"[WARN] reading {self.file_path}: {e}")
            return False

        if not new:
            return False

        any_added = False
        for raw in new.splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                for i, col_idx in enumerate(self.columns_to_read):
                    self.data[i].append(float(parts[col_idx]))
                any_added = True
            except (IndexError, ValueError):
                # Ignore malformed line
                continue

        # simple dt spike check (helps spot time read drift)
        try:
            t = self.data[0]
            if len(t) >= 3:
                dt_prev = t[-2] - t[-3]
                dt_cur  = t[-1] - t[-2]
                if dt_prev > 0 and dt_cur > 3.0 * dt_prev:
                    print(f"[WARN] large dt jump at t={t[-1]:.6f}: dt={dt_cur:.3e} (prev {dt_prev:.3e})")
        except Exception:
            pass

        return any_added


# ---------------------------
# Smoothing (phase-friendly)
# ---------------------------

def moving_average_same(x: np.ndarray, win: int) -> np.ndarray:
    """
    Zero-phase-ish MA with symmetric padding to keep the same length.
    """
    if win <= 1:
        return x.copy()
    if len(x) < 2:
        return x.copy()
    win = int(win)
    pad = win // 2
    # symmetric pad avoids initial lag
    xpad = np.pad(x, (pad, pad - (1 - win % 2)), mode="edge")
    kern = np.ones(win, dtype=float) / win
    y = np.convolve(xpad, kern, mode="valid")
    return y


def choose_window_by_seconds(t: np.ndarray, seconds: float, min_win: int = 5) -> int:
    """
    Convert a time window (seconds) to a sample window using median dt.
    """
    if len(t) < 3 or seconds <= 0:
        return max(1, min_win)
    dt = np.median(np.diff(t))
    if dt <= 0:
        return max(1, min_win)
    return max(min_win, int(round(seconds / dt)))


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # --- User knobs ---
    BASE_DIR = "../gymprecice-run"
    UPDATE_INTERVAL_SECONDS = 0.1
    SMOOTH_SECONDS = 5   # moving-average window in *seconds* for Cd/Cl plots
    SHOW_CL_COMPONENTS = True  # if True, try to also read Cl(f)/Cl(r) if present

    latest = get_latest_run_folder(BASE_DIR)
    if latest is None:
        print(f"[ERR] No folder with timestamp pattern under {BASE_DIR}")
        raise SystemExit(1)

    post_dir = Path(latest) / "env_0" / "fluid-openfoam" / "postProcessing"

    # --- Locate coefficient and control files (largest candidates) ---
    coeff_file = find_largest_matching(post_dir / "forceCoeffs", "coefficient.dat")
    # common locations: flowRateJet1/surfaceFieldValue.dat or similar
    try:
        control_file = find_largest_matching(post_dir / "flowRateJet1", "surfaceFieldValue")
    except FileNotFoundError:
        control_file = None

    print(f"[INFO] coefficients: {coeff_file}")
    print(f"[INFO] control     : {control_file if control_file is not None else '(not found)'}")

    # --- Detect Cd/Cl columns (0-based after Time) ---
    cd_idx, cl_idx = detect_force_coeff_columns(coeff_file)
    print(f"[INFO] detected columns -> Cd={cd_idx+1}, Cl={cl_idx+1} (1-based in file)")

    # Build the absolute column indices for the line split:
    # Time is column 0, then the detected indices are offset by +1
    cols = [0, 1 + cd_idx, 1 + cl_idx]  # [Time, Cd, Cl]

    # If you want Cl components too (Cl(f), Cl(r)) and they exist, try to find them quickly
    cl_f_idx = cl_r_idx = None
    if SHOW_CL_COMPONENTS:
        try:
            # reuse header tokenization; if present, locate 'cl(f)' and 'cl(r)'
            with open(coeff_file, "r") as f:
                hdr = []
                for _ in range(50):
                    ln = f.readline()
                    if not ln:
                        break
                    if ln.lstrip().startswith("#"):
                        hdr.append(ln.strip("# \n\t"))
                toks = " | ".join(hdr).lower()
                # crude indices: count tokens after first 'time'
                parts = re.split(r"[\s,;:|()\[\]\t]+", toks)
                parts = [p for p in parts if p]
                if "time" in parts:
                    start = parts.index("time") + 1
                    parts2 = parts[start:]
                    if "cl(f)" in parts2:
                        cl_f_idx = parts2.index("cl(f)")
                    if "cl(r)" in parts2:
                        cl_r_idx = parts2.index("cl(r)")
            if cl_f_idx is not None:
                cols.append(1 + cl_f_idx)
            if cl_r_idx is not None:
                cols.append(1 + cl_r_idx)
        except Exception:
            print("[WARN] could not print Cl components")
            pass

    # --- Monitors ---
    coeff_mon = DataMonitor(coeff_file, cols)
    if control_file is not None:
        ctrl_mon = DataMonitor(control_file, [0, 1])  # assume [time, value]
    else:
        ctrl_mon = None

    # --- Plot setup ---
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
    ax_cd, ax_cl, ax_u = axes

    (cd_raw_line,) = ax_cd.plot([], [], '-', lw=1.0, alpha=0.7, label='Cd')
    (cd_filt_line,) = ax_cd.plot([], [], '--', lw=1.2, label=f'Cd (MA {SMOOTH_SECONDS:.2f}s)')
    ax_cd.set_ylabel("Cd")
    ax_cd.grid(True, linestyle=":")
    ax_cd.legend(loc="upper right")

    (cl_raw_line,) = ax_cl.plot([], [], '-', lw=1.0, alpha=0.7, color='tab:green', label='Cl')
    (cl_filt_line,) = ax_cl.plot([], [], '--', lw=1.2, color='tab:purple', label=f'Cl (MA {SMOOTH_SECONDS:.2f}s)')
    if SHOW_CL_COMPONENTS and len(cols) >= 5:
        (cl_f_line,) = ax_cl.plot([], [], ':', lw=1.0, color='tab:red', label='Cl(f)')
        (cl_r_line,) = ax_cl.plot([], [], ':', lw=1.0, color='tab:blue', label='Cl(r)')
    ax_cl.set_ylabel("Cl")
    ax_cl.grid(True, linestyle=":")
    ax_cl.legend(loc="upper right")

    (u_line,) = ax_u.plot([], [], '-', lw=1.0, color='tab:orange', label='Control')
    ax_u.set_ylabel("u")
    ax_u.set_xlabel("Time")
    ax_u.grid(True, linestyle=":")
    ax_u.legend(loc="upper right")

    fig.suptitle(f"OpenFOAM forceCoeffs monitor\n{coeff_file}", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.show()

    print("\n[INFO] Live plotting…  (Ctrl+C to stop)")
    try:
        while plt.fignum_exists(fig.number):
            new_coeff = coeff_mon.read_new()
            new_ctrl = ctrl_mon.read_new() if ctrl_mon is not None else False

            if new_coeff or new_ctrl:
                # unpack
                time_c = np.asarray(coeff_mon.data[0], dtype=float)
                cd     = np.asarray(coeff_mon.data[1], dtype=float)
                cl     = np.asarray(coeff_mon.data[2], dtype=float)

                # choose MA window by seconds
                win = choose_window_by_seconds(time_c, SMOOTH_SECONDS, min_win=5)

                # filtered (same length as raw; symmetric padding to reduce phase lag)
                cd_f = moving_average_same(cd, win) if len(cd) else np.array([])
                cl_f = moving_average_same(cl, win) if len(cl) else np.array([])

                # update Cd plot
                cd_raw_line.set_data(time_c, cd)
                if len(cd_f) == len(cd):
                    cd_filt_line.set_data(time_c, cd_f)

                # update Cl plot
                cl_raw_line.set_data(time_c, cl)
                if len(cl_f) == len(cl):
                    cl_filt_line.set_data(time_c, cl_f)

                # optional Cl components
                if SHOW_CL_COMPONENTS and len(cols) >= 5 and len(coeff_mon.data) >= 5:
                    t_ok = time_c
                    if len(coeff_mon.data) >= 5 and len(coeff_mon.data[3]) == len(time_c):
                        clf = np.asarray(coeff_mon.data[3], dtype=float)
                        cl_f_line.set_data(t_ok, clf)
                    if len(coeff_mon.data) >= 6 and len(coeff_mon.data[4]) == len(time_c):
                        clr = np.asarray(coeff_mon.data[4], dtype=float)
                        cl_r_line.set_data(t_ok, clr)

                # control
                if ctrl_mon is not None:
                    t_u  = np.asarray(ctrl_mon.data[0], dtype=float)
                    uval = np.asarray(ctrl_mon.data[1], dtype=float)
                    u_line.set_data(t_u, uval)

                # rescale & redraw
                for ax in axes:
                    ax.relim()
                    # Apply autoscaling to all axes, BUT set custom limits for ax_cl
                    ax.autoscale_view()

                # *** MODIFICATION START ***
                # Manually adjust y-limits for ax_cl to ignore large spikes
                if len(cl) > 0:
                    # Filter out values outside a reasonable range for min/max calculation
                    # (e.g., anything beyond -5 to 5 for determining visual range,
                    # but allowing some margin for normal variations)
                    cl_display = cl[(cl > -5) & (cl < 5)]
                    if len(cl_display) > 0:
                        cl_min = np.min(cl_display)
                        cl_max = np.max(cl_display)
                        # Ensure there's a minimum visible range
                        margin = 0.5
                        ax_cl.set_ylim(min(cl_min - margin, -2.5), max(cl_max + margin, 2.5))
                    else:
                        # Fallback if no data within the filtered range
                        ax_cl.set_ylim(-2.5, 2.5)
                else:
                    ax_cl.set_ylim(-2.5, 2.5) # Default limits if no data yet
                # *** MODIFICATION END ***

                fig.canvas.draw()
                fig.canvas.flush_events()

            if not plt.fignum_exists(fig.number):  # extra guard mid-iteration
                break
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(UPDATE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        plt.ioff()  # turn off interactive
        plt.close('all')  # ensure window is closed; no blocking show

