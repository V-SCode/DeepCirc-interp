"""Figure S03 — Panel D: combined simplemotif rankings (CIRCUIT, log_odds).

Replaces the previous 4-stacked Da/Db/Dc/Dd panels with a single panel
containing all six rows (3 sizes × 2 directions) × 3 motifs per row =
18 motif cells. Direct paper-pipeline port of
``topology/figures/fig34v4_simplemotif_logodds_minimal.py``:

  - Same data source (combined k=3+4 pool, internal_only view,
    tier_pair=05, circuit task).
  - Same per-cell content: structural DAG + bold log_2 value below,
    green for +, red for −.
  - Same five hand-placed COORD_OVERRIDES (chain-on-horizontal layouts
    for the iso classes whose auto-layout otherwise routes one edge
    as a curve).
  - Paper ACCENT_BLUE single-color nodes (matches fig34v4's stripped
    minimal style — no source/inner/sink palette distinction here).

Outputs (vector + raster):
  panels/vector/panel_d.pdf | .svg
  panels/raster/panel_d.png

Run:
    python \\
        figures/setFinal/figS10/scripts/build_panel_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]   # .../DeepCirc-interp (repo root)
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Also expose topology/figures/ on sys.path so we can import
# fig34v4 + fig29 directly. This gives us the EXACT same edge-rendering
# logic fig34v4 uses (cluster fan-out, X-pattern arrowhead offsets,
# bezier-aware perimeter clipping) instead of re-implementing it.
_TOPO_FIGS = REPO_ROOT / "topology" / "figures"
if str(_TOPO_FIGS) not in sys.path:
    sys.path.insert(0, str(_TOPO_FIGS))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import DARK_GRAY, MID_GRAY, ACCENT_BLUE  # noqa: E402
# Import fig29's canonical edge renderer (carries cluster-fanout +
# X-pattern arrowhead offsets + bezier-aware perimeter clipping) and
# fig34v4's COORD_OVERRIDES + iso_key_to_adj + layered_layout. We do
# NOT use fig34v4.draw_skeleton_minimal directly because its
# bbox-aware node-radius auto-scaling makes longer-chain motifs render
# with smaller display nodes than the compact X-pattern motif in row
# 1 col 1 — we want uniform display size across ALL 18 cells. Solution
# below: pin every motif to a universal xlim/ylim and use a fixed
# node radius.
from fig29_simplemotif_by_size import (  # noqa: E402
    draw_edges_on_dag, iso_key_to_adj, layered_layout,
)
from fig34v4_simplemotif_logodds_minimal import COORD_OVERRIDES  # noqa: E402


# --- config ----------------------------------------------------------------

DATA = REPO_ROOT / "data" / "topology_g3" / "motif_tier_analysis"
PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS10"
              / "panels" / "vector")

TASK_DISK  = "circuit"
SIZES      = [5, 6, 7]            # 5/6/7-reg rows (4-reg dropped per fig34v4)
K_FILTER   = (3, 4)               # combined-k pool, k=5 excluded
TIER_PAIR  = "05"
N_PER_ROW  = 3

# Manifest slot: 96 × 184 mm (tightened from 126 $\\rightarrow$ 96 so Panels B and C
# can widen 60 $\\rightarrow$ 92 mm for design-space visuals).
CANVAS_W_MM = 136
CANVAS_H_MM = 184

# Visual style — paper ACCENT_BLUE for every node, all-black text,
# matches fig34v4's minimal look exactly.
NODE_COLOR  = "#18A8E8"           # paper ACCENT_BLUE
EDGE_COLOR  = DARK_GRAY
TEXT_BLACK  = "#111"
LOG2_POS    = ACCENT_BLUE         # paper signature cyan for + log₂
LOG2_NEG    = DARK_GRAY           # muted paper grey for − log₂

# Layout geometry. X_SPREAD widens the horizontal spacing between
# depth columns; bumped 1.5 $\\rightarrow$ 1.8 (vs fig34v4) to give the larger
# NODE_RADIUS=0.30 proper breathing room. COORD_OVERRIDES x-coords
# are scaled by the same 1.2× factor on retrieval.
X_SPREAD = 1.8
COORD_X_SCALE = X_SPREAD / 1.5   # = 1.2

# Universal data-coordinate bbox used for EVERY motif cell. Pinning
# all motifs to the same xlim/ylim gives a uniform display-per-data
# ratio across cells, so a fixed NODE_RADIUS produces uniform display
# node size. Tightened from (−0.6, 5.6) × (−1.5, 1.5) to use the cell
# area more effectively — the chain motif at X_SPREAD=1.8 spans 0–5.4
# horizontally and ±0 vertically; the worst-case off-line override
# node reaches y = ±1. Pad each direction by ~NODE_RADIUS for
# clearance against the cell border.
UNIV_XLIM = (-0.30, 5.70)
# Expanded from ±1.10 $\\rightarrow$ ±1.40 so the 3-fanout motif at y = ±1.0 no
# longer clips against the bbox once NODE_RADIUS=0.30 padding is
# applied. X is the aspect="equal" limiter (cell ~30×29mm with x
# range 6.0 vs y range 2.8 = 3:1 ratio of data-unit-per-pixel),
# so growing y just eats into the existing y padding inside the
# dag inset — node and edge DISPLAY size are unchanged.
UNIV_YLIM = (-1.40, 1.40)

# Fixed data-unit node radius. Display size = NODE_RADIUS / UNIV bbox
# width × inset display width. With X_SPREAD bumped to 1.8, the
# edge-length-to-node-diameter ratio is now 1.8 / 0.6 = 3.0 for chain
# motifs (was 2.5), giving proper visual spacing for the larger nodes.
NODE_RADIUS = 0.30


# --- skeleton renderer (uniform node size) --------------------------------

def draw_skeleton_uniform(ax, iso_key: str, k: int) -> None:
    """Draw a structural DAG with paper ACCENT_BLUE nodes, sized so the
    display node-radius is IDENTICAL across all motifs in the panel.

    Achieves uniformity by pinning every motif to the same xlim/ylim
    (UNIV_XLIM × UNIV_YLIM) and using a fixed NODE_RADIUS. The motif's
    actual layout coords are centred inside the universal bbox.

    Edge rendering delegates to fig29.draw_edges_on_dag, which carries
    the X-pattern arrowhead offsets, cluster fan-out, and bezier-aware
    perimeter clipping from the canonical renderer.
    """
    adj = iso_key_to_adj(iso_key, k)
    if iso_key in COORD_OVERRIDES:
        # Scale override x-coords by COORD_X_SCALE so their horizontal
        # spacing stays in sync with the bumped X_SPREAD. y unchanged
        # since override y-values (±1 for off-line nodes) are already
        # calibrated against the node radius for proper clearance.
        coords = {n: (x * COORD_X_SCALE, y)
                   for n, (x, y) in COORD_OVERRIDES[iso_key].items()}
    else:
        coords = layered_layout(adj)
        coords = {n: (x * X_SPREAD, y) for n, (x, y) in coords.items()}

    # Centre the motif inside the universal display box.
    xs = [coords[i][0] for i in range(k)]
    ys = [coords[i][1] for i in range(k)]
    motif_cx = (min(xs) + max(xs)) / 2.0
    motif_cy = (min(ys) + max(ys)) / 2.0
    box_cx = (UNIV_XLIM[0] + UNIV_XLIM[1]) / 2.0
    box_cy = (UNIV_YLIM[0] + UNIV_YLIM[1]) / 2.0
    dx_shift = box_cx - motif_cx
    dy_shift = box_cy - motif_cy
    coords = {n: (x + dx_shift, y + dy_shift) for n, (x, y) in coords.items()}

    ax.set_xlim(*UNIV_XLIM)
    ax.set_ylim(*UNIV_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    # Edges — fig29's canonical renderer (X-pattern offsets + cluster
    # fanout + bezier clipping). lw scaled down for the paper canvas.
    draw_edges_on_dag(ax, adj, coords, k,
                       node_radius=NODE_RADIUS,
                       edge_color=EDGE_COLOR,
                       lw=0.7)

    # Nodes — paper ACCENT_BLUE, fixed radius.
    for n in range(k):
        x, y = coords[n]
        ax.add_patch(plt.Circle((x, y), NODE_RADIUS,
                                  facecolor=NODE_COLOR,
                                  edgecolor=EDGE_COLOR,
                                  linewidth=0.3, zorder=3))


# --- data loading + ranking ------------------------------------------------

def load_combined_rankings() -> pd.DataFrame:
    frames = []
    for k in (3, 4):
        path = DATA / f"structural_iso_by_size_{TASK_DISK}_k{k}_internal_only.csv"
        df = pd.read_csv(path, dtype={"iso_key": str})
        df["iso_key"] = df["iso_key"].str.zfill(k * k)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def top_n_by(df: pd.DataFrame, col: str, *, ascending: bool,
              n: int = N_PER_ROW) -> pd.DataFrame:
    sub = df[~df["is_linear"]].copy()
    sub = sub.dropna(subset=[col])
    sub = sub.sort_values(col, ascending=ascending).head(n)
    return sub.reset_index(drop=True)


# --- panel assembly --------------------------------------------------------

def build_panel() -> None:
    use_style()

    df = load_combined_rankings()
    df = df[df["k"].isin(K_FILTER)]
    lo_col = f"log_odds_top{TIER_PAIR}_vs_bot{TIER_PAIR}"

    direction_specs = (
        ("Top 5%", False),
        ("Bot 5%", True),
    )

    n_rows = len(SIZES) * 2
    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))

    # Full one-line title — centred over the middle motif column.
    # Canvas widened so the full text fits at fontsize=8.
    fig.text(0.55, 0.985,
             "Most commonly occurring motifs by circuit size in Top and Bottom 5% of design space",
             ha="center", va="top",
             fontsize=9, fontweight="normal", color="#000000")

    # Tighten gridspec margins + spacing so each cell occupies a
    # larger fraction of the panel canvas — gives motifs more
    # display area without making the canvas bigger.
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=N_PER_ROW,
        hspace=0.06, wspace=0.02,
        # left bumped 0.07 $\\rightarrow$ 0.10 so the motif grid centres under the
        # title (title anchor sits at fig-x 0.55; old centre was 0.534,
        # new centre 0.549). Row labels at fig-x 0.04 still sit inside
        # the now-wider left margin.
        left=0.10, right=0.998, top=0.955, bottom=0.012,
    )

    row_idx = 0
    for s in SIZES:
        df_s = df[df["size_class"] == s]
        for direction_label, parent_asc in direction_specs:
            parents_row = top_n_by(df_s, lo_col, ascending=parent_asc)
            for col_idx in range(N_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                # Aggressively hide outer cell axes (xticks, yticks,
                # spines, grid all off — paper style enables grid by
                # default which leaks through ax.axis("off") alone).
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if col_idx >= len(parents_row):
                    continue
                parent = parents_row.iloc[col_idx]
                k = int(parent["k"])
                iso_key = parent["iso_key"]
                lo_val = float(parent[lo_col])

                # DAG fills almost the entire cell; Log_2 value sits
                # in a narrow band at the bottom.
                dag = ax.inset_axes([0.0, 0.10, 1.0, 0.89])
                dag.set_xticks([])
                dag.set_yticks([])
                dag.grid(False)
                for spine in dag.spines.values():
                    spine.set_visible(False)
                try:
                    # Uniform-size renderer: all motifs use the same
                    # data-coord bbox + fixed node radius so display
                    # size is identical across cells. Edge logic still
                    # comes from fig29.draw_edges_on_dag (preserves
                    # X-pattern arrowhead offsets + cluster fanout).
                    draw_skeleton_uniform(dag, iso_key, k)
                except Exception as e:
                    dag.text(0.5, 0.5, f"render err: {e!s:.16s}",
                              ha="center", va="center", fontsize=8)

                sign = "+" if lo_val >= 0 else ""
                color = LOG2_POS if lo_val >= 0 else LOG2_NEG
                # Use mathtext with \mathrm{Log} so the 'L' is uppercase
                # (default \log mathtext renders lowercase). Subscript
                # via mathtext so it renders on Arial (Arial doesn't
                # have the U+2082 \N{SUBSCRIPT TWO} glyph).
                ax.text(0.5, 0.09,
                          rf"$\mathrm{{Log}}_2$ = {sign}{lo_val:.2f}",
                          ha="center", va="center",
                          fontsize=8, fontweight="bold", color=color,
                          transform=ax.transAxes)

            # Row label upright in the top-left of each row, aligned
            # to the same x as the dashed separator line below
            # (gridspec left). Says "X regulator design space" rather
            # than "X regulators" so readers don't confuse the size
            # of the design space with the node count of the motif.
            label = f"{s} regulator design space - {direction_label}"
            bbox = gs[row_idx, 0].get_position(fig)
            fig.text(0.10, bbox.y1 - 0.005, label,
                      ha="left", va="top", rotation=0,
                      fontsize=8, fontweight="bold",
                      color=TEXT_BLACK)
            row_idx += 1

    # Horizontal dashed separators between the 5-REG / 6-REG / 7-REG
    # size groups. Each size occupies two consecutive rows (TOP-5%
    # then BOT-5%); separators sit mid-gap between row 1↔2 and 3↔4.
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    for i in (1, 3):
        y_sep = (bottoms[i] + tops[i + 1]) / 2
        fig.add_artist(Line2D(
            [lefts[0], rights[-1]], [y_sep, y_sep],
            transform=fig.transFigure,
            color=MID_GRAY, linewidth=0.5, linestyle=(0, (3, 3)),
        ))

    out_base = PANELS_VEC / "panel_d"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel D written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
