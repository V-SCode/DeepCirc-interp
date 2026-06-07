"""fig34v4 — fig34v3 data (top-3 log_odds, k=3+4) with fig35v3's minimal visual style.

Same six rows + three structural motifs per row as fig34v3 (top-3 by
within-size log_odds_top<TP>_vs_bot<TP> across the k=3 + k=4 pool —
HIGHEST for TOP rows, LOWEST for BOT rows). Visual styling lifted
from fig35v3:

  - Bigger DAGs, wider node spacing (X_SPREAD).
  - No n_topo / volume lines under each cell.
  - Corner badge: parent log₂ only (no k= suffix).
  - Single bold log₂ value below the DAG (green/red sign coloring).
  - Auto-scaled node radius so display size stays roughly consistent
    across chain / cluster / stacked motif shapes despite
    matplotlib's aspect="equal" behaviour.

Difference from fig35v3: the DAG shown here is the STRUCTURAL skeleton
(plain paper-blue nodes, no NIG part typing) — i.e., the parent
shape itself ranked by its own structural log_odds. fig35v3 swaps the
skeleton for the most-enriched NIG typing of that shape.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style as S
from fig29_simplemotif_by_size import (
    SIZE_CLASSES_DEFAULT,
    SIZE_COLOR,
    SIZE_LABEL,
    TASKS,
    TASK_TITLE,
    VIEW_SUFFIX,
    VIEW_TITLE_TAG,
    _draw_size_separators,
    _wrap_footer,
    draw_edges_on_dag,
    iso_key_to_adj,
    layered_layout,
    load_combined_rankings,
    top_n_by,
)


# Paper accent blue — matches fig29's structural-skeleton fill.
SKELETON_NODE_COLOR = "#18A8E8"

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "topology" / "figures" / "output" / "simplemotif" / "Path_1"

N_PER_ROW = 3
K_FILTER = (3, 4)
K_TAG = "k=3+4 (k=5 excluded)"
X_SPREAD = 1.5

# Hand-placed coordinates for specific iso classes where the auto-layout
# routes an edge as a curve around an intermediate node. Move the source
# node to a position that lets all edges draw as straight lines.
# Coords are in the same units layered_layout + X_SPREAD produces.
COORD_OVERRIDES: dict[str, dict[int, tuple[float, float]]] = {
    # 5-REG BOT-5% col 0, log₂ = -10.15. Candidate D from
    # _straight_arrow_experiment: collinear chain 2→3→0 placed on
    # horizontal y=0; node 1 dropped to (3, -1) so edge 2→1 and
    # edge 3→1 clear node 3 and node 0 respectively. All 4 edges
    # straight; longest chain reads as the visual backbone.
    "0000000001011100": {
        0: (3.00,  0.00),
        1: (3.00, -1.00),
        2: (0.00,  0.00),
        3: (1.50,  0.00),
    },
    # 5-REG TOP-5% col 1, log₂ = +1.61. Candidate-D-style mirror:
    # edges 1→3, 2→0, 3→0. Auto-layout bows edge 2→0 around node 3
    # at (1.5, 0). Place longest chain (1→3→0) on horizontal y=0;
    # drop off-chain source node 2 to (0, -1) so edge 2→0 is a clean
    # long diagonal that clears node 3 by 0.5 units.
    "0000000110001000": {
        0: (3.00,  0.00),
        1: (0.00,  0.00),
        2: (0.00, -1.00),
        3: (1.50,  0.00),
    },
    # 6-REG TOP-5% col 1, log₂ = +1.85. Edges 1→3, 2→0, 2→3, 3→0.
    # Same skip-edge pathology as the 5-REG TOP-5% col 1 motif
    # (2→0 bows around node 3); same Candidate-D mirror applies.
    # Extra edge 2→3 draws cleanly as a short up-right diagonal
    # from (0, -1) to (1.5, 0).
    "0000000110011000": {
        0: (3.00,  0.00),
        1: (0.00,  0.00),
        2: (0.00, -1.00),
        3: (1.50,  0.00),
    },
    # 6-REG TOP-5% col 0, log₂ = +2.24. Edges 1→3, 2→1, 2→3, 3→0.
    # Auto-layout zigzags the bottom chain 2→3→0 (y = 0, -0.55, 0).
    # Straighten the bottom 3 onto horizontal y=0; lift node 1 to
    # (0.75, +1.0) — between the first two bottom nodes — so both
    # 2→1 and 1→3 read as short symmetric diagonals (up-right then
    # down-right), avoiding any vertical stub. All 4 edges straight.
    "0000000101011000": {
        0: (3.00,  0.00),
        1: (0.75,  1.00),
        2: (0.00,  0.00),
        3: (1.50,  0.00),
    },
    # 6-REG TOP-5% col 2 + 7-REG TOP-5% col 2 (same iso class, two
    # rows share this override). log₂ = +1.34 and +1.00 respectively.
    # Edges 1→3, 2→0, 3→0, 3→2. Auto-layout zigzags the would-be
    # chain. Straighten chain 1→3→0 to horizontal y=0; lift node 2
    # to (2.25, +1.0) — between bottom nodes 3 and 0 since its
    # connections terminate there. Edges 3→2 and 2→0 become
    # symmetric short diagonals over node 0's side; all 4 straight.
    "0000000110001010": {
        0: (3.00,  0.00),
        1: (0.00,  0.00),
        2: (2.25,  1.00),
        3: (1.50,  0.00),
    },
    # 7-REG BOT-5% col 2, log₂ = -5.44. Same skeleton as
    # 0000000110001010 plus extra edge 1→2 (5 edges total). Same
    # layout principle: chain 1→3→0 on bottom, node 2 above at
    # (2.25, +1.0). Edge 1→2 becomes a long up-right diagonal that
    # clears node 3 at (1.5, 0) by 0.67 units. All 5 edges straight.
    "0000001110001010": {
        0: (3.00,  0.00),
        1: (0.00,  0.00),
        2: (2.25,  1.00),
        3: (1.50,  0.00),
    },
}


def draw_skeleton_minimal(ax, iso_key: str, k: int, *,
                            node_size: float = 0.30) -> None:
    """Plain paper-blue structural DAG with X_SPREAD + auto-scaled node
    radius — mirrors fig35v3's draw_typed_motif_flat but for the
    untyped structural skeleton."""
    adj = iso_key_to_adj(iso_key, k)
    if iso_key in COORD_OVERRIDES:
        coords = dict(COORD_OVERRIDES[iso_key])
    else:
        coords = layered_layout(adj)
        coords = {n: (x * X_SPREAD, y) for n, (x, y) in coords.items()}
    xs = [coords[i][0] for i in range(k)]
    ys = [coords[i][1] for i in range(k)]
    pad_x, pad_y = 0.6, 0.7
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    # Normalise display node size across motifs (mirrors fig35v3).
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    bbox_w = x_span + 2 * pad_x
    bbox_h = y_span + 2 * pad_y
    INSET_ASPECT = 2.0
    REF_PROXY = 5.7
    scale_proxy = max(bbox_w, bbox_h * INSET_ASPECT)
    node_radius = node_size * max(0.7, min(1.6, scale_proxy / REF_PROXY))
    draw_edges_on_dag(ax, adj, coords, k, node_radius=node_radius)
    for n in range(k):
        x, y = coords[n]
        ax.add_patch(plt.Circle((x, y), node_radius,
                                  facecolor=SKELETON_NODE_COLOR,
                                  edgecolor="#222",
                                  linewidth=0.7, zorder=3))


def render_skeleton_row(ax_grid, parent_rows: pd.DataFrame, *,
                          parent_logodds_col: str) -> None:
    n = len(parent_rows)
    if n == 0:
        ax_grid.text(0.5, 0.5, "(no non-linear parents pass filter)",
                       ha="center", va="center", transform=ax_grid.transAxes,
                       fontsize=8, color="#888", style="italic")
        ax_grid.axis("off")
        return
    ax_grid.axis("off")
    inner = ax_grid.inset_axes([0.02, 0.05, 0.96, 0.95])
    inner.axis("off")

    for i in range(n):
        sub = inner.inset_axes([i / n, 0.0, 1.0 / n, 1.0])
        sub.axis("off")
        sub.set_xlim(0, 1); sub.set_ylim(0, 1)

        parent = parent_rows.iloc[i]
        k = int(parent["k"])
        iso_key = parent["iso_key"]
        parent_lo = float(parent[parent_logodds_col])

        dag = sub.inset_axes([0.05, 0.22, 0.90, 0.74])
        try:
            draw_skeleton_minimal(dag, iso_key, k, node_size=0.2625)
        except Exception as e:
            dag.axis("off")
            dag.text(0.5, 0.5, f"render err: {e!s:.20s}",
                       ha="center", va="center", fontsize=6)

        sign = "+" if parent_lo >= 0 else ""
        color = "#117733" if parent_lo >= 0 else "#D7263D"
        sub.text(0.5, 0.10,
                  f"Log₂ = {sign}{parent_lo:.2f}",
                  ha="center", va="center",
                  fontsize=13.0, fontweight="bold", color=color)


def build_figure(task_disp: str, *,
                  motif_view: str = "internal_only",
                  sizes: list[int] | None = None,
                  min_n_topo: int = 1,
                  tier_pair: str = "05") -> None:
    if tier_pair not in ("01", "05", "10", "25"):
        raise ValueError(f"tier_pair must be '01'|'05'|'10'|'25'; got {tier_pair!r}")
    task_disk = TASKS[task_disp]
    sizes = list(sizes) if sizes else list(SIZE_CLASSES_DEFAULT)

    df = load_combined_rankings(task_disk, motif_view=motif_view)
    df = df[df["k"].isin(K_FILTER)]
    if min_n_topo > 1:
        df = df[df["n_topo_with_motif_in_size"] >= min_n_topo]

    tier_pct = {"01": "1%", "05": "5%", "10": "10%", "25": "25%"}[tier_pair]
    lo_col = f"log_odds_top{tier_pair}_vs_bot{tier_pair}"

    direction_specs = (
        ("TOP-" + tier_pct, False),
        ("BOT-" + tier_pct, True),
    )

    n_rows = len(sizes) * 2
    fig_h = 3.4 * n_rows + 0.8
    fig = plt.figure(figsize=(17.0, fig_h))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.22,
                            left=0.07, right=0.98, top=0.93, bottom=0.06)

    row_idx = 0
    for s in sizes:
        df_s = df[df["size_class"] == s]
        for direction_label, parent_asc in direction_specs:
            parents_row = top_n_by(df_s, lo_col,
                                     ascending=parent_asc, n=N_PER_ROW)
            ax = fig.add_subplot(gs[row_idx, 0])
            render_skeleton_row(ax, parents_row,
                                  parent_logodds_col=lo_col)
            label = f"{SIZE_LABEL[s]}\n{direction_label}"
            fig.text(0.015, 0.93 - (row_idx + 0.5) * (0.87 / n_rows),
                      label, ha="left", va="center", rotation=90,
                      fontsize=14, fontweight="bold", color="#111")
            row_idx += 1

    _draw_size_separators(fig, gs, len(sizes))

    fig.text(0.5, 0.955,
              "TOP 3 BY LOG₂ ENRICHMENT  ·  most differential motifs (good vs bad)",
              ha="center", va="bottom",
              fontsize=11.0, fontweight="bold", color="#111")

    S.figure_title(fig,
        f"Simplemotif by SIZE × DIRECTION  ·  {TASK_TITLE[task_disp]}  ·  "
        f"{K_TAG}  ·  TIER = top-{tier_pct} vs bot-{tier_pct}"
        f"{VIEW_TITLE_TAG[motif_view]}")

    footer_parts = [
        f"Each row = (size class × direction). Top-3 non-linear iso-"
        f"canonical motifs by within-size log₂(top-{tier_pct}_S / "
        f"bot-{tier_pct}_S) across the k=3 + k=4 pool (k=5 excluded) — "
        f"TOP rows show the most elite-enriched structural shapes "
        f"(highest +log₂); BOT rows show the most failure-enriched "
        f"shapes (lowest −log₂). Each cell carries its structural DAG + "
        f"a single bold log₂ value (green if elite-driven, red if "
        f"failure-driven). Corner badge repeats the same log₂ for at-a-"
        f"glance scanning. See fig35v3 for the matching figure where "
        f"each skeleton is replaced by its most-enriched NIG typing.",
    ]
    if motif_view == "internal_only":
        footer_parts.append(
            "Internal-only view: IN sensors and OUT-* terminal gates "
            "excluded; only internal NOT + NOR participate.")
    elif motif_view == "no_in":
        footer_parts.append(
            "No-IN view: sensor inputs excluded; OUT-* gates retained.")
    elif motif_view == "no_out":
        footer_parts.append(
            "No-OUT view: OUT-* terminal gates excluded; IN sensors "
            "retained (symmetric counterpart to no_in).")
    fig.text(0.5, 0.02,
              _wrap_footer("  ·  ".join(footer_parts), width=260),
              ha="center", va="bottom", fontsize=9, color="#555",
              style="italic")

    view_tag = VIEW_SUFFIX[motif_view]
    tier_tag = f"_top{tier_pair}_bot{tier_pair}"
    name = f"fig34v4_logodds_minimal_{task_disp}{view_tag}{tier_tag}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTDIR / f"{name}.pdf"
    png_path = OUTDIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  rendered {pdf_path.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="circuit",
                     choices=["all", "circuit", "growth"])
    ap.add_argument("--motif_view",
                     choices=list(VIEW_SUFFIX.keys()), default="internal_only")
    ap.add_argument("--sizes", type=int, nargs="+",
                     default=list(SIZE_CLASSES_DEFAULT))
    ap.add_argument("--min_n_topo", type=int, default=1)
    ap.add_argument("--tier_pair", choices=["01", "05", "10", "25"],
                     default="05")
    args = ap.parse_args()
    S.setup_matplotlib()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for t in tasks:
        build_figure(t,
                     motif_view=args.motif_view,
                     sizes=args.sizes,
                     min_n_topo=args.min_n_topo,
                     tier_pair=args.tier_pair)


if __name__ == "__main__":
    main()
