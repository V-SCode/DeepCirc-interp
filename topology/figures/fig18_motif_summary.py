"""fig18 — Topological subgraph motif analysis summary (k=3, k=4, k=5).

Consolidates the motif story across all three subgraph sizes into one
summary slide. Each row tells one story arc across k=3 -> k=4 -> k=5
so the recurring patterns are obvious at a glance:

  Row 1: BEST circuit motif — constructive position-diverse cascade
         ending in OUT-NOT. Partial r +0.38 / +0.36 / +0.34.
  Row 2: WORST circuit motif — output-cone catastrophe with two+
         NOR-out nodes converging on OUT-OR2 output. -0.48 / -0.47 / -0.29.
  Row 3: BEST toxicity motif — input-proximal protective pattern with
         NOR-in clustering and OUT-OR2. +0.27 / +0.28 / +0.27.
  Row 4: WORST toxicity motif — mid-NOR clustering ending in OUT-NOT.
         Same pattern that's BEST for circuit (cross-task tension).
         -0.26 / -0.27 / -0.27.

Bottom: principles + cross-task tension callout.

Data source: data/G3/l3/l3_motif_correlations_typed{,_k4,_k5}.csv.
Phase / Script: P13 (motif features at k=3, 4, 5; v2.0 position-aware
typing + OUT-gate semantic split).

Method: P13 enumerates connected k-node subgraph motifs in each of 215
trained topology DAGs. Each node typed via depth-aware v2.0 typing
(NOR-in/-mid/-out, NOT-in/-mid/-out, OUT-NOT/-OR2/-OR3/-OR4+). For each
typed motif: counts per topology -> Pearson + partial-r (controlling
for num_edges) against circuit_log_max and toxicity_max. Universal
threshold: n_topo ≥ 5 + |partial_r| ≥ 0.2.

Usage:
    python topology/figures/fig18_motif_summary.py --group G3
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style as S
from _loaders import load_motif_correlations


# ---------------------------------------------------------------------------
# Node colors / labels (mirror fig05_2)
# ---------------------------------------------------------------------------

NODE_COLORS = {
    "IN":       "#92C5DE",
    "NOT":      "#F4A582",
    "NOT-in":   "#F4A582",
    "NOT-mid":  "#F4A582",
    "NOT-out":  "#F4A582",
    "NOR":      "#5AAE61",
    "NOR-in":   "#A6D96A",
    "NOR-mid":  "#5AAE61",
    "NOR-out":  "#1A9850",
    "OUT":      "#404040",
    "OUT-NOT":  "#7B3294",
    "OUT-OR2":  "#404040",
    "OUT-OR3":  "#404040",
    "OUT-OR4+": "#404040",
    "?":        "#BBBBBB",
}

NODE_LABELS = {
    "IN":  "IN",  "NOT": "NOT", "NOT-in": "NOT", "NOT-mid": "NOT", "NOT-out": "NOT",
    "NOR": "NOR", "NOR-in": "NOR", "NOR-mid": "NOR", "NOR-out": "NOR",
    "OUT": "OUT", "OUT-NOT": "OUT", "OUT-OR2": "OUT", "OUT-OR3": "OUT", "OUT-OR4+": "OUT",
    "?":   "?",
}


# ---------------------------------------------------------------------------
# Motif parsing + layout + drawing
# ---------------------------------------------------------------------------

def parse_motif(motif_str: str) -> tuple[list[str], np.ndarray, int]:
    types_part, edges_part = motif_str.split("|edges=")
    tokens = types_part.split("/")
    types = [t.split("(")[0] for t in tokens]
    k = len(types)
    A = np.array([int(c) for c in edges_part]).reshape(k, k)
    return types, A, k


def compute_layout(types: list[str], A: np.ndarray, k: int) -> list[tuple[float, float]]:
    G = nx.DiGraph()
    for i in range(k):
        G.add_node(i, type=types[i])
    for i in range(k):
        for j in range(k):
            if A[i, j]:
                G.add_edge(i, j)
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not sources:
        return [(i, 0.0) for i in range(k)]
    depth = {s: 0 for s in sources}
    try:
        for n in nx.topological_sort(G):
            for s in G.successors(n):
                depth[s] = max(depth.get(s, 0), depth.get(n, 0) + 1)
    except nx.NetworkXUnfeasible:
        return [(i, 0.0) for i in range(k)]
    by_depth: dict[int, list[int]] = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)
    coords: list[tuple[float, float] | None] = [None] * k
    for d, ns in by_depth.items():
        ns_sorted = sorted(ns, key=lambda n: (types[n], -G.in_degree(n)))
        n_at_depth = len(ns_sorted)
        for i, n in enumerate(ns_sorted):
            x = d
            y = (n_at_depth - 1) / 2.0 - i
            coords[n] = (x, y)

    # Linear-look fix: when every depth column has exactly one node, the
    # backbone collapses to a single y=0 row and skip edges arc over it.
    # Zigzag the intermediate columns so the chain spreads vertically and
    # skip edges can route as straight diagonals (handled by the obstacle-
    # aware curve check in _arrow).
    max_depth = max(depth.values()) if depth else 0
    if max_depth >= 2 and all(
        len(by_depth.get(d, [])) == 1 for d in range(max_depth + 1)
    ):
        has_skip = any(
            A[i, j] and abs(depth[j] - depth[i]) >= 2
            for i in range(k) for j in range(k)
        )
        if has_skip:
            zigzag_offset = 0.55
            for d in range(1, max_depth):
                n = by_depth[d][0]
                cur = coords[n]
                if cur is None:
                    continue
                y_new = zigzag_offset if (d % 2 == 1) else -zigzag_offset
                coords[n] = (cur[0], y_new)
    return [c if c is not None else (0.0, 0.0) for c in coords]


def _arrow(ax, p1, p2, radius=0.20, *, other_coords=None):
    """Draw a directed arrow from p1 to p2.

    If ``other_coords`` is provided (the full list of node positions in
    the motif), the edge curves only when the straight line would clip
    one of those intermediate nodes, and the bow direction is chosen to
    pass AWAY from the obstacle. This gives clean straight diagonals
    over zigzagged single-occupancy chains and only bows when needed.
    Without ``other_coords`` the function falls back to the legacy
    span-based bow rule (for callers that haven't been updated).
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    sx, sy = x1 + ux * radius, y1 + uy * radius
    ex, ey = x2 - ux * radius * 1.05, y2 - uy * radius * 1.05

    rad = 0.0
    if other_coords is not None:
        obstacle_clearance = radius * 1.3
        for c in other_coords:
            if c is None:
                continue
            xn, yn = c
            if (abs(xn - x1) < 1e-9 and abs(yn - y1) < 1e-9):
                continue
            if (abs(xn - x2) < 1e-9 and abs(yn - y2) < 1e-9):
                continue
            if not (min(x1, x2) < xn < max(x1, x2)):
                continue
            if abs(x2 - x1) > 1e-9:
                t = (xn - x1) / (x2 - x1)
            else:
                t = 0.0
            y_line = y1 + t * (y2 - y1)
            if abs(yn - y_line) < obstacle_clearance:
                rad_mag = 0.10 + 0.04 * abs(dx)
                cross = (x2 - x1) * (yn - y1) - (y2 - y1) * (xn - x1)
                bow_sign = -1.0 if cross > 0 else 1.0
                rad = bow_sign * rad_mag
                break
    else:
        span_x = abs(dx)
        if span_x >= 1.5:
            rad_mag = 0.10 + 0.04 * span_x
            rad = rad_mag if (y1 + y2) >= 0 else -rad_mag

    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.2,
                                shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad:.3f}"),
                zorder=2)


def draw_motif(ax, types: list[str], A: np.ndarray, k: int,
                *, node_size: float = 0.22, label_size: float = 7.5) -> None:
    coords = compute_layout(types, A, k)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x, pad_y = 0.6, 0.7
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(k):
        for j in range(k):
            if A[i, j]:
                _arrow(ax, coords[i], coords[j], radius=node_size,
                       other_coords=[coords[n] for n in range(k)
                                       if n != i and n != j])
    for n in range(k):
        x, y = coords[n]
        face = NODE_COLORS.get(types[n], "#bbb")
        circle = plt.Circle((x, y), node_size, facecolor=face,
                             edgecolor="#333", linewidth=0.8, zorder=4)
        ax.add_patch(circle)
        label_text = NODE_LABELS.get(types[n], types[n])
        is_out = types[n].startswith("OUT")
        ax.text(x, y, label_text, ha="center", va="center",
                fontsize=label_size,
                color="white" if is_out else "#111",
                fontweight="bold", zorder=5)


def top_motif(df: pd.DataFrame, *, score: str, direction: str,
              min_topo: int = 10) -> pd.Series:
    """Return the single strongest motif. Falls back to min_topo=5 if
    min_topo=10 yields nothing."""
    for m in (min_topo, 5):
        sub = df[(df["score"] == score) & (df["n_topologies"] >= m)].copy()
        if direction == "good":
            sub = sub[sub["partial_r"] > 0]
            if not sub.empty:
                return sub.nlargest(1, "partial_r").iloc[0]
        else:
            sub = sub[sub["partial_r"] < 0]
            if not sub.empty:
                return sub.nsmallest(1, "partial_r").iloc[0]
    raise ValueError(f"no motif found for {score}/{direction}")


# ---------------------------------------------------------------------------
# Row configuration
# ---------------------------------------------------------------------------

ROWS = [
    {
        "title": "BEST CIRCUIT motif",
        "subtitle": "constructive cascade ending in OUT-NOT",
        "score": "circuit_log_max",
        "direction": "good",
        "color": "#117733",
        "color_band": "#e8f4ea",
    },
    {
        "title": "WORST CIRCUIT motif",
        "subtitle": "output-cone catastrophe — NOR-out clustering -> OUT-OR2",
        "score": "circuit_log_max",
        "direction": "bad",
        "color": "#882255",
        "color_band": "#f5e8ec",
    },
    {
        "title": "BEST TOXICITY motif",
        "subtitle": "input-proximal pattern — IN / NOR-in feed OUT-OR2",
        "score": "toxicity_max",
        "direction": "good",
        "color": "#0072B2",
        "color_band": "#e2eef7",
    },
    {
        "title": "WORST TOXICITY motif",
        "subtitle": "mid-NOR clustering -> OUT-NOT (= BEST for circuit)",
        "score": "toxicity_max",
        "direction": "bad",
        "color": "#D55E00",
        "color_band": "#f7e6dc",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default="G3")
    args = ap.parse_args()

    S.setup_matplotlib()

    # Load typed correlations per k
    dfs = {3: load_motif_correlations(args.group, "typed", k=3),
           4: load_motif_correlations(args.group, "typed", k=4),
           5: load_motif_correlations(args.group, "typed", k=5)}

    # k stats
    k_stats = {}
    for k_val, df in dfs.items():
        k_stats[k_val] = {
            "n_kept": df["motif"].nunique(),
            "n_circ": int(((df.score == "circuit_log_max") &
                           (df.n_topologies >= 5)).sum()),
        }

    fig = plt.figure(figsize=(19.0, 15.0))

    # Outer grid: 6 rows (header, 4 motif rows, footer) × 4 cols
    # (1 row-title col + 3 k-cols)
    outer = fig.add_gridspec(
        nrows=6, ncols=4,
        height_ratios=[0.40, 1, 1, 1, 1, 1.05],
        width_ratios=[0.55, 1, 1, 1],
        hspace=0.16, wspace=0.05,
        left=0.02, right=0.98, top=0.96, bottom=0.02,
    )

    # ============================================================
    # Header — title row spanning all columns
    # ============================================================
    ax_h = fig.add_subplot(outer[0, :])
    ax_h.set_xlim(0, 100)
    ax_h.set_ylim(0, 100)
    ax_h.axis("off")
    ax_h.text(50, 85,
              "Topological subgraph motifs at k = 3, 4, 5   ·   "
              "215 trained topologies",
              ha="center", va="center", fontsize=S.TITLE_SIZE + 6,
              fontweight="bold")
    ax_h.text(50, 50,
              "What does the GAT learn at the local-subgraph level? "
              "Each row reads left-to-right (k=3 -> k=4 -> k=5). "
              "Same pattern emerges at every scale.",
              ha="center", va="center", fontsize=10,
              style="italic", color="#555")

    # k column header labels (just above the motif grid)
    for j, k_val in enumerate((3, 4, 5)):
        stats = k_stats[k_val]
        ax_kh = fig.add_subplot(outer[0, j + 1])
        ax_kh.set_xlim(0, 100)
        ax_kh.set_ylim(0, 100)
        ax_kh.axis("off")
        ax_kh.text(50, 18, f"k = {k_val}",
                   ha="center", va="bottom",
                   fontsize=S.TITLE_SIZE + 3, fontweight="bold",
                   color="#222")
        ax_kh.text(50, 4,
                   f"{stats['n_kept']:,} typed motifs kept",
                   ha="center", va="bottom", fontsize=8.5, color="#777")

    # ============================================================
    # Four motif rows
    # ============================================================
    for row_idx, row_cfg in enumerate(ROWS):
        # Row title column — left side of row
        ax_label = fig.add_subplot(outer[row_idx + 1, 0])
        ax_label.set_xlim(0, 100)
        ax_label.set_ylim(0, 100)
        ax_label.axis("off")

        # Background band for the entire row (across all 4 columns).
        # We need a band that visually spans cols 0-3. Use a figure-level
        # rectangle via fig.patches, or place a same-color FancyBboxPatch
        # in each column-axis. Simpler: draw the band in the label-column
        # axis with a wide x-range and rely on outer column borders to
        # match visually.
        ax_label.add_patch(FancyBboxPatch(
            (5, 8), 95, 84,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            fc=row_cfg["color_band"], ec=row_cfg["color"], lw=1.2,
            alpha=0.95, zorder=0,
        ))
        ax_label.text(50, 70, row_cfg["title"],
                       ha="center", va="center", fontsize=11.5,
                       fontweight="bold", color=row_cfg["color"],
                       wrap=True)
        # Wrap subtitle
        sub_wrapped = textwrap.fill(row_cfg["subtitle"], width=22)
        ax_label.text(50, 38, sub_wrapped,
                       ha="center", va="center", fontsize=8.5,
                       style="italic", color=row_cfg["color"])

        # Three k columns
        for j, k_val in enumerate((3, 4, 5)):
            ax_cell = fig.add_subplot(outer[row_idx + 1, j + 1])
            # Background band same color as label cell
            ax_cell.add_patch(FancyBboxPatch(
                (0.5, 8), 99, 84,
                boxstyle="round,pad=0.5,rounding_size=1.2",
                fc=row_cfg["color_band"], ec=row_cfg["color"], lw=1.0,
                alpha=0.95, zorder=0,
                transform=ax_cell.transAxes,
            ))
            ax_cell.set_xlim(0, 100)
            ax_cell.set_ylim(0, 100)
            ax_cell.axis("off")

            try:
                m = top_motif(dfs[k_val],
                               score=row_cfg["score"],
                               direction=row_cfg["direction"])
            except ValueError:
                ax_cell.text(50, 50, "no eligible motif",
                              ha="center", va="center",
                              fontsize=10, color="#999")
                continue
            types, A, k = parse_motif(m["motif"])

            # Inner axes for the DAG inside the cell
            # Use ax_cell.inset_axes for proper nesting.
            ax_dag = ax_cell.inset_axes([0.05, 0.30, 0.90, 0.55])
            draw_motif(ax_dag, types, A, k,
                        node_size=0.26, label_size=8.0)

            # Type label above DAG (compact)
            type_label = " / ".join(types)
            if len(type_label) > 42:
                type_label = type_label[:42] + "..."
            ax_cell.text(50, 90, type_label,
                          ha="center", va="top", fontsize=7.5,
                          color="#666", family="monospace")

            # partial r + n_topo annotations below DAG
            ax_cell.text(50, 22, f"partial r = {m['partial_r']:+.3f}",
                          ha="center", va="center", fontsize=11,
                          fontweight="bold", color=row_cfg["color"])
            ax_cell.text(50, 12, f"n_topo = {int(m['n_topologies'])}",
                          ha="center", va="center", fontsize=9,
                          color="#444")

    # ============================================================
    # Footer — principles + cross-task tension
    # ============================================================
    ax_f = fig.add_subplot(outer[5, :])
    ax_f.set_xlim(0, 100)
    ax_f.set_ylim(0, 100)
    ax_f.axis("off")

    ax_f.add_patch(FancyBboxPatch(
        (0.5, 5), 99, 90,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        fc="#2c3e50", ec="none", alpha=0.96, zorder=1,
    ))

    ax_f.text(50, 90, "PRINCIPLES + CROSS-TASK TENSION",
              ha="center", va="center", fontsize=11,
              color="#ecf0f1", alpha=0.85, fontweight="bold")

    principles = [
        ("A.  POSITION-DIVERSITY > NODE TYPE",
         "At every k, the strongest circuit-positive motif has NOR cascade with mixed -in/-mid/-out depth labels; the strongest circuit-negative has all-NOR-out clustering. Same edge structure, opposite verdicts — POSITION decides."),
        ("B.  NOT-BUFFER TERMINUS IS CONSTRUCTIVE",
         "Every BEST circuit motif at k=3/4/5 ends in OUT-NOT (single-promoter NOT-output). Every WORST ends in OUT-OR2 (tandem-promoter OR). The v2.0 OUT-gate semantic split is load-bearing for the circuit dichotomy."),
        ("C.  EFFECT MAGNITUDE DECAYS WITH k",
         "k=3 partial r magnitudes reach ±0.48; k=4 ±0.47; k=5 ±0.29. Larger motifs have more variables that dilute the signal, but the SIGN is preserved — principles transfer cleanly k=3 -> k=4 -> k=5."),
        ("D.  CROSS-TASK TENSION IS REAL",
         "The motif that's BEST for circuit (NOR-mid/NOR-out/OUT-NOT at k=3, partial r +0.38) is the SAME pattern that's WORST for toxicity (partial r -0.26 at k=3). Same physical structure, opposite biological outcome. Pareto trade-off lives at the motif level."),
    ]
    for i, (head, body) in enumerate(principles):
        col = i % 2
        row = i // 2
        x = 1.5 + col * 49.5
        y = 75 - row * 36
        ax_f.text(x + 1, y, head, ha="left", va="top",
                   fontsize=10, fontweight="bold", color="#FDCB6E")
        body_wrapped = textwrap.fill(body, width=66)
        ax_f.text(x + 1, y - 8, body_wrapped, ha="left", va="top",
                   fontsize=8.3, color="#ecf0f1")

    pdf_path, png_path = S.save_figure(fig, "fig18_motif_summary", args.group)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
