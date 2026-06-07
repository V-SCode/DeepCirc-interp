"""Figure S12 — per-NPN Pareto fronts (3 x 3 grid of all 9 NPN classes).

Standalone figure under the setFinal/ layout (previously figS04 Panel B
in the setB_design_arc landscape composite). Now expanded to cover ALL
nine NPN equivalence classes represented in the G3 substrate rather
than just the top-3 most-populated, since the standalone canvas
affords the extra room.

Cells in topology-count descending order across rows (left-to-right,
top-to-bottom):
  Row 1: 0x03 (60), 0x17 (51), 0x0F (25)
  Row 2: 0x1B (24), 0x07 (16), 0x3C (15)
  Row 3: 0x19 (10), 0x06 (9),  0x16 (5)

Per-cell content:
  - Pareto fronts: one PCHIP-smoothed curve per topology, coloured by
    target hex.
  - Knee markers (one per topology, same target colour).
  - Cell-top label naming the NPN class + its topology count.
  - Per-cell legend listing contributing target hexes.

Outputs (vector + raster):
  panels/vector/panel_b.pdf | .svg
  panels/raster/panel_b.png

Run:
    python \\
        figures/setFinal/figS12/scripts/build_panel_b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "topology" / "figures"))
import _loaders as L  # noqa: E402
import _style as S    # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import DARK_GRAY  # noqa: E402


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS12"
              / "panels" / "vector")

TEXT_GREY = "#666666"


# --- canvas + 3 x 3 grid geometry ------------------------------------------

# Standalone figure: 3 x 3 grid of NPN cells. Width matches figS13 (306 mm)
# for consistent width across the setFinal cross-topology figures; height
# 320 mm gives each cell ~95 x 88 mm, comparable to the old single-row
# Panel B's per-cell size.
CANVAS_W_MM   = 360
CANVAS_H_MM   = 360
LEFT_PAD_MM   = 10.0
RIGHT_PAD_MM  = 4.0
TOP_PAD_MM    = 20.0   # below figure-level title
BOTTOM_PAD_MM = 14.0   # above figure bottom
H_GAP_MM      = 8.0    # horizontal gap between cells
V_GAP_MM      = 14.0   # vertical gap between cell rows (cell-top label needs room)

N_COLS = 3
N_ROWS = 3
CELL_W_MM = (CANVAS_W_MM - LEFT_PAD_MM - RIGHT_PAD_MM
              - (N_COLS - 1) * H_GAP_MM) / N_COLS
CELL_H_MM = (CANVAS_H_MM - TOP_PAD_MM - BOTTOM_PAD_MM
              - (N_ROWS - 1) * V_GAP_MM) / N_ROWS


def _cell_position(row: int, col: int) -> tuple[float, float]:
    """Bottom-left (x_mm, y_mm) of the cell at (row, col).

    Row 0 is the TOP row; matplotlib's coordinate origin is bottom-left,
    so we flip row index when computing y.
    """
    x_pitch = CELL_W_MM + H_GAP_MM
    y_pitch = CELL_H_MM + V_GAP_MM
    x = LEFT_PAD_MM + col * x_pitch
    # Top row sits at the top of the usable region.
    top_row_y = CANVAS_H_MM - TOP_PAD_MM - CELL_H_MM
    y = top_row_y - row * y_pitch
    return (x, y)


def _to_axes_rect(x_mm: float, y_mm: float) -> tuple[float, float, float, float]:
    return (
        x_mm / CANVAS_W_MM,
        y_mm / CANVAS_H_MM,
        CELL_W_MM / CANVAS_W_MM,
        CELL_H_MM / CANVAS_H_MM,
    )


# --- per-NPN-class color overrides -----------------------------------------

# Only NPN classes whose target colours need adjustment for visual
# consistency get overrides; all others fall back to S.TARGET_COLORS.

NPN_0X17_COLOR_OVERRIDE = {
    # The two paper anchor targets (0x17, 0x2B) get the saturated
    # ACCENT_BLUE pair so blue dominates at first glance; the other
    # four targets sit in distinct PASTEL hue families.
    "0x17": "#18A8E8",   # ACCENT_BLUE
    "0x2B": "#0C7AB0",   # ACCENT_BLUE_DARK
    "0x4D": "#C5A1D5",   # PASTEL purple
    "0xE8": "#BFD9AE",   # PASTEL green
    "0x71": "#E6B37A",   # PASTEL orange
    "0x8E": "#E7A3A3",   # PASTEL pink
}

NPN_0X0F_COLOR_OVERRIDE = {
    # 0x0F: replaces magenta with paper PASTEL blue.
    "0x0F": "#9EC7E8",
}

NPN_0X16_COLOR_OVERRIDE = {
    # 0x16 carries the 0x6D paper anchor; keep ACCENT_BLUE on the anchor.
    "0x6D": "#18A8E8",
}

NPN_COLOR_OVERRIDES = {
    "0x17": NPN_0X17_COLOR_OVERRIDE,
    "0x0F": NPN_0X0F_COLOR_OVERRIDE,
    "0x16": NPN_0X16_COLOR_OVERRIDE,
}


def _color_for(target: str, npn: str) -> str:
    override = NPN_COLOR_OVERRIDES.get(npn)
    if override and target in override:
        return override[target]
    return S.TARGET_COLORS.get(target, "#999999")


# --- main ------------------------------------------------------------------

def build_panel() -> None:
    use_style()

    knees   = L.load_pareto_knees("G3")
    fronts  = L.load_pareto_fronts("G3")

    knees["npn_class"]     = knees["source"].apply(S.npn_class_of)
    fronts["npn_class"]    = fronts["source"].apply(S.npn_class_of)
    fronts["target_label"] = fronts["source"].apply(S.parse_source)
    knees["target_label"]  = knees["source"].apply(S.parse_source)

    # ALL NPN classes sorted by topology count (descending).
    topo_counts = knees.groupby("npn_class").size().sort_values(ascending=False)
    npn_classes_to_show = topo_counts.index.tolist()
    n_cells = N_ROWS * N_COLS
    if len(npn_classes_to_show) > n_cells:
        npn_classes_to_show = npn_classes_to_show[:n_cells]

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))

    # Shared axis limits across all 9 cells for direct visual comparison.
    xlo = min(knees["knee_A_circuit_log"].min() - 0.3, -1.5)
    xhi = max(knees["knee_A_circuit_log"].max() + 0.5, 14.5)
    ylo, yhi = 0.65, 1.0

    for idx, npn in enumerate(npn_classes_to_show):
        row, col = divmod(idx, N_COLS)
        x_mm, y_mm = _cell_position(row, col)
        ax = fig.add_axes(_to_axes_rect(x_mm, y_mm))

        sub_fronts = fronts[fronts["npn_class"] == npn]
        sub_knees  = knees[knees["npn_class"] == npn]
        targets    = sorted(sub_knees["target_label"].unique())
        n_topo     = topo_counts[npn]

        # --- Pareto fronts (one PCHIP curve per topology) ---------------
        for tid, grp in sub_fronts.groupby("topology_id"):
            target = grp["target_label"].iloc[0]
            color = _color_for(target, npn)
            grp_sorted = grp.sort_values("circuit_log")
            x = grp_sorted["circuit_log"].to_numpy()
            y = grp_sorted["toxicity"].to_numpy()
            if len(x) >= 2:
                y_rev = y[::-1]
                suffix_max = np.maximum.accumulate(y_rev)[::-1]
                keep = y >= suffix_max
                x = x[keep]
                y = y[keep]
                _, uniq_idx = np.unique(x, return_index=True)
                uniq_idx = np.sort(uniq_idx)
                x = x[uniq_idx]
                y = y[uniq_idx]
            if len(x) >= 4:
                pchip = PchipInterpolator(x, y, extrapolate=False)
                x_smooth = np.linspace(x[0], x[-1], 120)
                y_smooth = pchip(x_smooth)
                ax.plot(x_smooth, y_smooth, color=color,
                         alpha=0.60, linewidth=0.7,
                         zorder=3, solid_capstyle="round")
            elif len(x) >= 2:
                ax.plot(x, y, color=color, alpha=0.60, linewidth=0.7,
                         zorder=3, solid_capstyle="round")

        # --- Knee markers (one per topology) ----------------------------
        for target in targets:
            sub_t = sub_knees[sub_knees["target_label"] == target]
            ax.scatter(
                sub_t["knee_A_circuit_log"], sub_t["knee_A_toxicity"],
                s=10, c=_color_for(target, npn),
                edgecolor="white", linewidth=0.25, zorder=5,
            )

        # --- Cell-top label: NPN class + topology count -----------------
        ax.text(
            0.02, 1.02,
            f"NPN {npn} ,  n = {n_topo} topologies",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color="#000000", fontweight="normal",
        )

        # --- Per-cell target legend -------------------------------------
        legend_handles = [
            Line2D([], [], marker="o", color="none",
                    markerfacecolor=_color_for(t, npn),
                    markeredgecolor="white", markersize=4, label=t)
            for t in targets
        ]
        leg = ax.legend(
            handles=legend_handles, loc="lower left",
            frameon=False, fontsize=12,
            handletextpad=0.25, columnspacing=0.5,
            labelspacing=0.18, borderpad=0.05,
            ncol=2 if len(targets) > 3 else 1,
        )
        for txt in leg.get_texts():
            txt.set_color("#000000")

        # --- Cell styling ----------------------------------------------
        ax.grid(True, linewidth=0.25, alpha=0.35,
                 color="#000000", zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(DARK_GRAY)
            ax.spines[spine].set_linewidth(0.6)
        ax.tick_params(
            axis="both", which="major",
            color=DARK_GRAY, width=0.6, length=2.5,
            labelsize=12, labelcolor="#000000",
        )

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        # x-label only on bottom-row cells; y-label only on left-column cells.
        if row == N_ROWS - 1:
            ax.set_xlabel(r"Circuit Score (log$_2$)",
                           fontsize=12, color="#000000", labelpad=1.5)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Growth Score (raw)",
                           fontsize=12, color="#000000", labelpad=1.5)
        else:
            ax.set_yticklabels([])

    # --- Figure-level title ------------------------------------------------
    fig.text(0.5, 0.985, "Pareto fronts per NPN class",
              ha="center", va="top",
              fontsize=16, fontweight="normal", color="#000000")

    out_base = PANELS_VEC / "panel_b"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel B (3 x 3 NPN grid) written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
