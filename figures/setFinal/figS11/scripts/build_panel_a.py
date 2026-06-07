"""Figure S04 — Panel A: most-enriched NIG typing per parent shape.

Paper-pipeline port of ``topology/figures/output/simplemotif/
Path_1/fig35v3_most_enriched_typed_minimal_circuit_internal_only_top05_bot05``.
Companion to figS03 Panel D (which ports fig34v4 — the same parent
shapes but with plain paper-blue skeletons). Panel A shows the SAME 18
parent shapes (3 sizes × 2 directions × 3 motifs) replaced by their
most-enriched NIG-typed variant: which NOT/NOR composition drives that
shape into the elite or failure tier the hardest.

Each cell:
  - Typed-motif DAG in paper PASTEL gate palette (green for NOR-*,
    orange for NOT-*, dark grey for OUT-*, purple for OUT-NOT)
  - Parent-skeleton log₂ chip in the upper-left of the cell
  - Bold typed-variant log₂ value below the DAG (green for +, red for −)

Uniform display size across all 18 cells via universal bbox + fixed
node radius (same approach as figS03 Panel D). Edge rendering uses
fig29.draw_edges_on_dag so motifs are bit-identical to fig35v3:
X-pattern arrowhead offsets, cluster fan-out, bezier-aware perimeter
clipping. Typed-motif node indices aligned to parent skeleton via
fig30._align_typed_to_parent so override coords (which key by parent
iso) line up cleanly.

Outputs (vector + raster):
  panels/vector/panel_a.pdf | .svg
  panels/raster/panel_a.png

Run:
    python \\
        figures/setFinal/figS11/scripts/build_panel_a.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]   # .../DeepCirc-interp (repo root)
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Expose topology/figures/ so we can import the canonical
# rendering helpers + override dict.
_TOPO_FIGS = REPO_ROOT / "topology" / "figures"
if str(_TOPO_FIGS) not in sys.path:
    sys.path.insert(0, str(_TOPO_FIGS))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import PASTEL, DARK_GRAY, MID_GRAY, ACCENT_BLUE  # noqa: E402

# Canonical edge renderer + layered layout + iso decoder from fig29.
from fig29_simplemotif_by_size import (  # noqa: E402
    draw_edges_on_dag, iso_key_to_adj, layered_layout,
    load_combined_rankings, top_n_by,
)
# Typed-motif data: per-size typed expansions + log_odds normalisation.
from fig30_simplemotif_by_size_typed import (  # noqa: E402
    DATA as TYPED_DATA_DIR,
    _align_typed_to_parent,
    _attach_typed_log_odds,
    _size_tier_totals,
)
# Hand-placed coords for the 5 iso classes that otherwise route a
# curved edge (shared with fig34v4 / fig35v3 / figS03 Panel D).
from fig34v4_simplemotif_logodds_minimal import COORD_OVERRIDES  # noqa: E402
# Motif string parser (gives types[], adj_matrix, k).
from fig18_motif_summary import parse_motif  # noqa: E402


# --- config ----------------------------------------------------------------

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS11"
              / "panels" / "vector")

TASK_DISK = "circuit"
SIZES     = [5, 6, 7]
K_FILTER  = (3, 4)
TIER_PAIR = "05"
N_PER_ROW = 3

# Canvas at fig35v3's native ~0.80 aspect (17 × 21.2 in $\\rightarrow$ 432 × 538 mm
# at native scale; here scaled down for paper).
CANVAS_W_MM = 240
CANVAS_H_MM = 300

# Universal data-coordinate bbox + fixed node radius. Pinning every
# motif to the same xlim/ylim makes display node size uniform across
# all 18 cells. UNIV bbox expanded ~5% vs figS03 Panel D so motifs
# render slightly smaller relative to their cells, and UNIV_YLIM is
# wider (-1.40 $\\rightarrow$ 1.40) to fit the vertical span of 3-node fan-out
# motifs (e.g., the 7-REG BOT col 0 motif with nodes at y = ±1)
# without clipping at the cell edges.
UNIV_XLIM   = (-0.45, 5.85)
UNIV_YLIM   = (-1.40, 1.40)
NODE_RADIUS = 0.30
X_SPREAD    = 1.8
COORD_X_SCALE = X_SPREAD / 1.5

# Paper PASTEL gate palette. Translates fig35v3's
# FLAT_NODE_COLORS (#88CC85 NOR / #F4A582 NOT / #404040 OUT /
# #7B3294 OUT-NOT / #92C5DE IN) onto paper hues.
#
# Two-step monochrome scale for NOR ↔ NOT (2026-06-01): NOR is the
# lighter mid-grey, NOT is the darker mid-grey. Both flat across
# depths. Keeps OUT at DARK_GRAY so the ordinal NOR (light) $\\rightarrow$ NOT
# (mid) $\\rightarrow$ OUT (dark) reads as a 3-step grey progression.
NOR_GRAY = "#D4D4D4"   # light-mid grey for NOR depths
NOT_GRAY = "#646464"   # mid-dark grey for NOT depths
GATE_COLOR = {
    "NOR":     NOR_GRAY,
    "NOR-in":  NOR_GRAY,
    "NOR-mid": NOR_GRAY,
    "NOR-out": NOR_GRAY,
    "NOT":     NOT_GRAY,
    "NOT-in":  NOT_GRAY,
    "NOT-mid": NOT_GRAY,
    "NOT-out": NOT_GRAY,
    # OUT terminal types
    "OUT":      DARK_GRAY,
    "OUT-NOT":  PASTEL["purple"],
    "OUT-OR2":  DARK_GRAY,
    "OUT-OR3":  DARK_GRAY,
    "OUT-OR4+": DARK_GRAY,
    # IN sensor
    "IN":       PASTEL["blue"],
    # Untyped fallback
    "?":        "#BBBBBB",
}
EDGE_COLOR = DARK_GRAY
LOG2_POS   = ACCENT_BLUE   # paper signature cyan for + log₂
LOG2_NEG   = DARK_GRAY     # muted paper grey for − log₂


# --- typed motif renderer --------------------------------------------------

def draw_typed_motif_pastel(ax, motif_str: str,
                              parent_iso_key: str | None = None) -> None:
    """NIG-typed DAG with paper PASTEL gate palette + uniform node size.

    Aligns the typed motif to the parent skeleton (so COORD_OVERRIDES
    keyed by parent iso line up node-for-node), then renders edges via
    fig29.draw_edges_on_dag (carries cluster fanout + X-pattern offsets
    + bezier-aware perimeter clipping).
    """
    types, A, k = parse_motif(motif_str)
    if parent_iso_key is not None:
        types, A = _align_typed_to_parent(types, A, parent_iso_key, k)

    if parent_iso_key in COORD_OVERRIDES:
        coords = {n: (x * COORD_X_SCALE, y)
                   for n, (x, y) in COORD_OVERRIDES[parent_iso_key].items()}
    else:
        coords = layered_layout(A)
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Edges via canonical renderer (X-pattern offsets etc. preserved).
    draw_edges_on_dag(ax, A, coords, k,
                       node_radius=NODE_RADIUS,
                       edge_color=EDGE_COLOR,
                       lw=0.7)

    # Nodes with paper PASTEL gate colors.
    for n in range(k):
        x, y = coords[n]
        face = GATE_COLOR.get(types[n], "#BBBBBB")
        ax.add_patch(plt.Circle((x, y), NODE_RADIUS,
                                  facecolor=face,
                                  edgecolor=EDGE_COLOR,
                                  linewidth=0.3, zorder=3))


# --- data loading ----------------------------------------------------------

def load_typed_df() -> pd.DataFrame:
    """Load typed-expansion table for circuit + internal_only view."""
    suffix = "_internal_only"
    typed_df = pd.read_csv(
        TYPED_DATA_DIR / f"typed_expansions_by_size_{TASK_DISK}{suffix}.csv",
        dtype={"parent_iso_key": str},
    )
    typed_df["parent_iso_key"] = typed_df.apply(
        lambda r: str(r["parent_iso_key"]).zfill(int(r["k"]) ** 2),
        axis=1)
    tier_totals = _size_tier_totals(TASK_DISK, tier_pair=TIER_PAIR)
    typed_df = _attach_typed_log_odds(typed_df, tier_totals,
                                        tier_pair=TIER_PAIR)
    return typed_df


def best_typed_variant(typed_df: pd.DataFrame, *,
                         k: int, parent_iso_key: str, size_class: int,
                         logodds_col: str,
                         ascending: bool) -> pd.Series | None:
    matches = typed_df[
        (typed_df["k"] == k)
        & (typed_df["parent_iso_key"] == parent_iso_key)
        & (typed_df["size_class"] == size_class)
    ].dropna(subset=[logodds_col])
    if len(matches) == 0:
        return None
    return matches.sort_values(logodds_col, ascending=ascending).iloc[0]


# --- panel assembly --------------------------------------------------------

def build_panel() -> None:
    use_style()

    parent_df = load_combined_rankings(TASK_DISK, motif_view="internal_only")
    parent_df = parent_df[parent_df["k"].isin(K_FILTER)]
    parent_lo = f"log_odds_top{TIER_PAIR}_vs_bot{TIER_PAIR}"

    typed_df = load_typed_df()
    typed_lo = f"log_odds_top{TIER_PAIR}_vs_bot{TIER_PAIR}_S"

    direction_specs = (
        ("Top 5%", False, False),   # parent ascending, typed ascending
        ("Bot 5%", True,  True),
    )

    n_rows = len(SIZES) * 2
    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))

    # Panel title — DeepCirc paper formatting.
    fig.text(0.535, 0.985,
              "Most-enriched NOT/NOR gate composition per motif across circuit sizes",
              ha="center", va="top",
              fontsize=11, fontweight="normal", color="#000000")

    # Tight gridspec margins so motifs fill the canvas.
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=N_PER_ROW,
        hspace=0.08, wspace=0.03,
        left=0.07, right=0.997, top=0.965, bottom=0.012,
    )

    row_idx = 0
    for s in SIZES:
        df_s = parent_df[parent_df["size_class"] == s]
        for direction_label, parent_asc, typed_asc in direction_specs:
            parents_row = top_n_by(df_s, parent_lo,
                                     ascending=parent_asc, n=N_PER_ROW)
            for col_idx in range(N_PER_ROW):
                ax = fig.add_subplot(gs[row_idx, col_idx])
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
                parent_log = float(parent[parent_lo])

                # DAG fills most of cell; Log_2 value sits at bottom.
                dag = ax.inset_axes([0.0, 0.10, 1.0, 0.89])

                best = best_typed_variant(
                    typed_df, k=k, parent_iso_key=iso_key, size_class=s,
                    logodds_col=typed_lo, ascending=typed_asc,
                )
                if best is None:
                    dag.text(0.5, 0.5, "(no typed variants)",
                              ha="center", va="center", fontsize=10,
                              color="#888", style="italic",
                              transform=dag.transAxes)
                    continue

                try:
                    draw_typed_motif_pastel(
                        dag, best["typed_motif"],
                        parent_iso_key=iso_key,
                    )
                except Exception as e:
                    dag.text(0.5, 0.5, f"render err: {e!s:.16s}",
                              ha="center", va="center", fontsize=10,
                              transform=dag.transAxes)

                # Bold typed Log₂ value below the DAG, with Parent
                # Log₂ stacked below it. Both at the same fontsize
                # (7.5pt); typed Log₂ is bold + green/red, Parent
                # Log₂ is regular + grey so it reads as supporting
                # context.
                typed_log = float(best[typed_lo])
                sign_t = "+" if typed_log >= 0 else ""
                color = LOG2_POS if typed_log >= 0 else LOG2_NEG
                ax.text(0.5, 0.12,
                          rf"$\mathrm{{Log}}_2$ = {sign_t}{typed_log:.2f}",
                          ha="center", va="center",
                          fontsize=10, fontweight="bold", color=color,
                          transform=ax.transAxes)
                sign_p = "+" if parent_log >= 0 else ""
                ax.text(0.5, 0.03,
                          rf"$\mathrm{{Parent\ Log}}_2$ = {sign_p}{parent_log:.2f}",
                          ha="center", va="center",
                          fontsize=10, color="#666",
                          transform=ax.transAxes)

            # Row label upright in the top-left margin of each row
            # (replaces the previous rotation=90 sideways label on
            # the far left edge). Anchored just inside the grid
            # margin at the top of each row's bbox.
            label = f"{s} regulator design space - {direction_label}"
            bbox = gs[row_idx, 0].get_position(fig)
            fig.text(0.07, bbox.y1 - 0.005, label,
                      ha="left", va="top", rotation=0,
                      fontsize=10, fontweight="bold", color="#111")
            row_idx += 1

    # Horizontal dashed separators between the 5-REG / 6-REG / 7-REG
    # size groups. Each size occupies two consecutive rows (TOP-5%
    # then BOT-5%); separators sit mid-gap between row 1↔2 and 3↔4.
    bottoms, tops, lefts, rights = gs.get_grid_positions(fig)
    for i in (1, 3):
        y_sep = (bottoms[i] + tops[i + 1]) / 2
        # Slightly darker than MID_GRAY (#9E9E9E) so the separator
        # renders at a similar visual weight to figS03 Panel D's
        # separators after the composer scales this 240$\\rightarrow$280 mm
        # canvas up ~1.17×.
        fig.add_artist(Line2D(
            [lefts[0], rights[-1]], [y_sep, y_sep],
            transform=fig.transFigure,
            color="#666666", linewidth=0.6, linestyle=(0, (3, 3)),
        ))

    out_base = PANELS_VEC / "panel_a"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel A (figS04) written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
