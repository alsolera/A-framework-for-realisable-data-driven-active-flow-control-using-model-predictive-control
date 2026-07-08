"""
plot_cp_surface_distribution.py

Aerodynamic-style Cp distribution over the truck surface.
Convention: Cp negative → curve goes AWAY from the truck (suction)
            Cp positive → curve goes INTO the truck (pressure)

Surfaces:
  - Top side    : probes  0–39, y = +0.5, projected in +y
  - Bottom side : probes 40–79, y = −0.5, projected in −y
  - Rear face   : probes 80–89, x = 7.647, projected in +x
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from pathlib import Path

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

uncontrolled_probe_file = Path(
    "../gymprecice-run/jet_2Dtruck_DNSv3_20250831_1241_noAction/"
    "env_0/fluid-openfoam/postProcessing/probes/0/p"
)
controlled_probe_file = Path(
    "../gymprecice-run/jet_2Dtruck_DNSv3_20250906_19_38_58_SlimMPC_20250906_223800/"
    "env_0/fluid-openfoam/postProcessing/probes/0/p"
)

output_dir  = Path("../04_Results/paper_figures/")
output_name = "cp_surface_distribution"

# Physical constants (OpenFOAM non-dimensional)
rho          = 1.0
U_inf        = 1.0
p_inf        = 0.0
dyn_pressure = 0.5 * rho * U_inf ** 2

start_time_controlled   = 100.0
start_time_uncontrolled = 0.0

# Visual scale: truck-units per unit of Cp
CP_SCALE = 0.8

# ---------------------------------------------------------------------------
# PROBE COORDINATES
# ---------------------------------------------------------------------------

probe_coords = {
     0: (0.301634,  0.5),  1: (0.485268,  0.5),  2: (0.668902,  0.5),
     3: (0.852537,  0.5),  4: (1.036171,  0.5),  5: (1.219805,  0.5),
     6: (1.403439,  0.5),  7: (1.587073,  0.5),  8: (1.770707,  0.5),
     9: (1.954341,  0.5), 10: (2.137976,  0.5), 11: (2.321610,  0.5),
    12: (2.505244,  0.5), 13: (2.688878,  0.5), 14: (2.872512,  0.5),
    15: (3.056146,  0.5), 16: (3.239780,  0.5), 17: (3.423415,  0.5),
    18: (3.607049,  0.5), 19: (3.790683,  0.5), 20: (3.974317,  0.5),
    21: (4.157951,  0.5), 22: (4.341585,  0.5), 23: (4.525220,  0.5),
    24: (4.708854,  0.5), 25: (4.892488,  0.5), 26: (5.076122,  0.5),
    27: (5.259756,  0.5), 28: (5.443390,  0.5), 29: (5.627024,  0.5),
    30: (5.810659,  0.5), 31: (5.994293,  0.5), 32: (6.177927,  0.5),
    33: (6.361561,  0.5), 34: (6.545195,  0.5), 35: (6.728829,  0.5),
    36: (6.912463,  0.5), 37: (7.096098,  0.5), 38: (7.279732,  0.5),
    39: (7.463366,  0.5),
    40: (0.301634, -0.5), 41: (0.485268, -0.5), 42: (0.668902, -0.5),
    43: (0.852537, -0.5), 44: (1.036171, -0.5), 45: (1.219805, -0.5),
    46: (1.403439, -0.5), 47: (1.587073, -0.5), 48: (1.770707, -0.5),
    49: (1.954341, -0.5), 50: (2.137976, -0.5), 51: (2.321610, -0.5),
    52: (2.505244, -0.5), 53: (2.688878, -0.5), 54: (2.872512, -0.5),
    55: (3.056146, -0.5), 56: (3.239780, -0.5), 57: (3.423415, -0.5),
    58: (3.607049, -0.5), 59: (3.790683, -0.5), 60: (3.974317, -0.5),
    61: (4.157951, -0.5), 62: (4.341585, -0.5), 63: (4.525220, -0.5),
    64: (4.708854, -0.5), 65: (4.892488, -0.5), 66: (5.076122, -0.5),
    67: (5.259756, -0.5), 68: (5.443390, -0.5), 69: (5.627024, -0.5),
    70: (5.810659, -0.5), 71: (5.994293, -0.5), 72: (6.177927, -0.5),
    73: (6.361561, -0.5), 74: (6.545195, -0.5), 75: (6.728829, -0.5),
    76: (6.912463, -0.5), 77: (7.096098, -0.5), 78: (7.279732, -0.5),
    79: (7.463366, -0.5),
    80: (7.647,  0.368182), 81: (7.647,  0.286364), 82: (7.647,  0.204545),
    83: (7.647,  0.122727), 84: (7.647,  0.040909), 85: (7.647, -0.040909),
    86: (7.647, -0.122727), 87: (7.647, -0.204545), 88: (7.647, -0.286364),
    89: (7.647, -0.368182),
}

n_probes = 90
probe_xy = np.array([probe_coords[i] for i in range(n_probes)])

# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

def parse_probes(filepath, start_time=0.0):
    filepath = Path(filepath)
    if filepath.is_dir():
        filepath = filepath / 'p'
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            rows.append(s.split())
    data = np.array(rows, dtype=float)
    mask = data[:, 0] >= start_time
    if not mask.any():
        raise ValueError(f"No data after t={start_time} in {filepath}")
    cp = (data[mask, 1:] - p_inf) / dyn_pressure
    return cp.mean(axis=0)

# ---------------------------------------------------------------------------
# TRUCK OUTLINE
# ---------------------------------------------------------------------------

def make_truck_patch(**kw):
    rectwidth  = 7.65
    rectheight = 1.0
    radius     = 0.118

    a_top = np.linspace(np.pi / 2, np.pi, 20)
    ax_t  = radius + radius * np.cos(a_top)
    ay_t  = (rectheight / 2 - radius) + radius * np.sin(a_top)

    a_bot = np.linspace(np.pi, 3 * np.pi / 2, 20)
    ax_b  = radius + radius * np.cos(a_bot)
    ay_b  = (-rectheight / 2 + radius) + radius * np.sin(a_bot)

    verts = (
        [(rectwidth, rectheight / 2), (radius, rectheight / 2)]
        + list(zip(ax_t, ay_t))
        + [(0, -rectheight / 2 + radius)]
        + list(zip(ax_b, ay_b))
        + [(rectwidth, -rectheight / 2), (rectwidth, rectheight / 2)]
    )
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    return PathPatch(MplPath(verts, codes), **kw)

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_cp_aero(cp_uc, cp_c, probe_xy, scale, output_path):

    COLOR_UC = 'k'
    COLOR_C  = 'tab:blue'
    LW       = 1.3

    rectwidth  = 7.65
    rectheight = 1.0

    fig, ax = plt.subplots(figsize=(5, 3))

    # --- Truck body (on top of fills, below curves) ---
    ax.add_patch(make_truck_patch(
        facecolor='white', edgecolor='black', lw=1.2, zorder=10))

    # ------------------------------------------------------------------ #
    def draw_side(indices, outward_sign, along_col, wall_col,
                  label_uc=None, label_c=None):
        s    = probe_xy[indices, along_col]
        wall = probe_xy[indices[0], wall_col]

        srt   = np.argsort(s)
        s     = s[srt]
        uc    = cp_uc[indices][srt]
        c     = cp_c [indices][srt]

        off_uc = -uc * scale * outward_sign
        off_c  = -c  * scale * outward_sign

        if wall_col == 1:      # top / bottom: project in y
            y_wall = np.full_like(s, wall)
            y_uc   = wall + off_uc
            y_c    = wall + off_c

            # Draw UC fill first, then C fill on top — no blending artifacts
            ax.fill_between(s, y_wall, y_uc,
                            color=COLOR_UC, alpha=0.15, zorder=3, lw=0)
            ax.fill_between(s, y_wall, y_c,
                            color=COLOR_C,  alpha=0.15, zorder=4, lw=0)

            ax.plot(s, y_uc, color=COLOR_UC, lw=LW, label=label_uc,
                    solid_capstyle='round', zorder=6)
            ax.plot(s, y_c,  color=COLOR_C,  lw=LW, label=label_c,
                    solid_capstyle='round', zorder=7)

        else:                  # rear: project in x
            x_wall = np.full_like(s, wall)
            x_uc   = wall + off_uc
            x_c    = wall + off_c

            ax.fill_betweenx(s, x_wall, x_uc,
                             color=COLOR_UC, alpha=0.15, zorder=3, lw=0)
            ax.fill_betweenx(s, x_wall, x_c,
                             color=COLOR_C,  alpha=0.15, zorder=4, lw=0)

            ax.plot(x_uc, s, color=COLOR_UC, lw=LW, label=label_uc,
                    solid_capstyle='round', zorder=6)
            ax.plot(x_c,  s, color=COLOR_C,  lw=LW, label=label_c,
                    solid_capstyle='round', zorder=7)
    # ------------------------------------------------------------------ #

    # Top side (probes 0-39)
    draw_side(np.arange(0,  40), outward_sign=+1, along_col=0, wall_col=1,
              label_uc='Uncontrolled', label_c='Controlled')

    # Bottom side (probes 40-79)
    draw_side(np.arange(40, 80), outward_sign=-1, along_col=0, wall_col=1)

    # Rear face (probes 80-89)
    draw_side(np.arange(80, 90), outward_sign=+1, along_col=1, wall_col=0)

    # --- Sensor location dots (subtle, on the wall) ---
    ax.scatter(probe_xy[:40,   0], np.full(40,  rectheight/2), s=4,
               color='grey', zorder=11, lw=0)
    ax.scatter(probe_xy[40:80, 0], np.full(40, -rectheight/2), s=4,
               color='grey', zorder=11, lw=0)
    ax.scatter(np.full(10, 7.647), probe_xy[80:90, 1], s=4,
               color='grey', zorder=11, lw=0)

    # --- Cp = 0 dashed reference at each wall ---
    x0_side = probe_xy[0,  0]
    x1_side = probe_xy[39, 0]
    ax.plot([x0_side, x1_side], [ rectheight/2,  rectheight/2],
            color='#aaaaaa', lw=0.5, ls=':', zorder=2)
    ax.plot([x0_side, x1_side], [-rectheight/2, -rectheight/2],
            color='#aaaaaa', lw=0.5, ls=':', zorder=2)
    y0_rear = probe_xy[89, 1]
    y1_rear = probe_xy[80, 1]
    ax.plot([rectwidth, rectwidth], [y0_rear, y1_rear],
            color='#aaaaaa', lw=0.5, ls=':', zorder=2)

    # --- Scale bar: placed below bottom-right of the truck ---
    bar_cp  = 1
    bar_len = bar_cp * scale          # length in truck-units
    bar_x0  = 5.5                     # start x (centred on rear half)
    bar_y   = rectheight/2 + 1   # below the bottom curve

    # horizontal double-headed arrow
    ax.annotate('', xy=(bar_x0 + bar_len, bar_y), xytext=(bar_x0, bar_y),
                arrowprops=dict(arrowstyle='<->', color='#555555',
                                lw=0.8, mutation_scale=6))
    # vertical tick marks at each end
    tick_h = 0.04
    for bx in [bar_x0, bar_x0 + bar_len]:
        ax.plot([bx, bx], [bar_y - tick_h, bar_y + tick_h],
                color='#555555', lw=0.8, zorder=8)
    ax.text(bar_x0 + bar_len / 2, bar_y - 0.20,
            rf'$\Delta C_p = {bar_cp}$',
            ha='center', va='top', color='#333333')

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(),
              loc='upper left', bbox_to_anchor=(0, 1.2))

    # --- Axes ---
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.4, rectwidth + 1)
    #ax.set_ylim(-rectheight/2 - 1.40, rectheight/2 + 1.30)
    ax.axis('off')

    plt.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.pdf'):
        fig.savefig(output_path.with_suffix(ext), bbox_inches='tight')
        print(f"Saved: {output_path.with_suffix(ext)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Parsing uncontrolled case ...")
    cp_uc = parse_probes(uncontrolled_probe_file,
                         start_time=start_time_uncontrolled)

    print("Parsing controlled case (t >= 100) ...")
    cp_c  = parse_probes(controlled_probe_file,
                         start_time=start_time_controlled)

    print("Plotting ...")
    plot_cp_aero(cp_uc, cp_c, probe_xy,
                 scale=CP_SCALE,
                 output_path=output_dir / output_name)
    print("Done.")