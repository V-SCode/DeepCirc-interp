"""Figure S04 — Panel C: knee-vs-max-circuit design-space comparison.

Promoted 2026-05-27 from panel_draft_c_knee_vs_maxcirc_with_yd to be the
canonical Panel C of figS04. The previous Panel C (per-target best by
circuit_log, no knee cluster) is preserved at panel_c_draft.{pdf,svg,png}
for provenance.

Layout (280×200 mm — matches the figS04 manifest slot):

  LEFT pane (scatter):
    * 215 LIGHT BLUE dots   — each topology's best Pareto KNEE design
                              (knee_A_circuit_log, knee_A_toxicity).
    * 215 LIGHT PURPLE dots — each topology's MAX-CIRCUIT design
                              (highest circuit_log among that topology's
                              buildable Pareto-front designs at the
                              paper MLP gates: circuit_score >= 2 AND
                              growth >= 0.5).
    * 3 GOLD STARS          — paper Fig. 3 yellow-dots (0x2B / 0x17 /
                              0x6D), simulator-rank-1 reference designs.
    * Buildable region shaded pastel green, threshold floors + grid all
      LIGHT_GRAY for uniform background.

  RIGHT pane:
    * 30-row table = TOP 15 BY MAX-CIRCUIT (per target) + TOP 15 BY KNEE
      (per target). Both groups deduplicated to 1 design per target
      function so cross-target diversity is preserved.
    * Knee top-15 ranked by combined_score recomputed across all 215
      knees at the PAPER MLP-gate floors (not the Path-A subset that
      cross_target_portfolio.csv pre-filters).
    * Yellow-dot reference table at the bottom (3 rows, simulator
      ground-truth scores).
    * Bottom legend strip naming the 5 most-frequent parts (PhlF/P2,
      SrpR/S4, AmtR/A1, BetI/E1, QacR/Q2) — colour-coded for in-row
      scanning.

  Cross-pane signal flow within each part list: parts are reordered by
  graph position (input-proximal $\\rightarrow$ output-adjacent via d2o BFS over the
  topology DAG). PhlF/P2 sits at the rightmost (output-adjacent) slot
  in nearly every best design — a spatial trend the per-row sort
  exposes directly.

Outputs:
  panels/vector/panel_c.{pdf,svg}
  panels/raster/panel_c.png
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "topology" / "figures"))
import _loaders as L  # noqa: E402
import _style as S    # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    ACCENT_YELLOW, ACCENT_YELLOW_DARK,
    DARK_GRAY, MID_GRAY, LIGHT_GRAY, PASTEL,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS13"
              / "panels" / "vector")

TEXT_GREY = "#666666"
PASS_GREEN = "#1A9850"
FAIL_RED   = "#D73027"

# Paper MLP-gate thresholds (from upstream NNGGA pipeline L1546/L1553).
CIRC_SCORE_FLOOR = 2.0
CIRC_LOG_FLOOR   = float(np.log2(CIRC_SCORE_FLOOR))  # = 1.0 in log₂
TOX_FLOOR        = 0.5

# Top-5 parts across the n=20 per-target best designs. Each gets a
# distinct paper-consistent hue so a reader can scan rows for "where
# does part X appear?". Order = descending presence count.
TOP5_COLORS = {
    # Paper-aligned family colours from Extended Data Fig. 4 (Option B
    # — paper colours exactly, with BetI saturation-bumped from the
    # paper's very-pale #D0E0F0 to a more readable #90B5D0 since the
    # table cells need higher legibility than the gene-arrow shapes).
    "PhlF/P2": "#F0B070",   # paper PhlF orange/peach
    "SrpR/S4": "#5080C0",   # paper SrpR medium blue
    "AmtR/A1": "#90C080",   # paper AmtR light green
    "BetI/E1": "#90B5D0",   # paper BetI (saturation-bumped pale blue)
    "QacR/Q2": "#E06070",   # paper QacR red/rose
}

# 20-part TetR-family library; index ↔ name mapping (paper Methods).
LIBRARY_PART_NAMES = {
    0:  "AmeR/F1",   1:  "AmtR/A1",
    2:  "BetI/E1",
    3:  "BM3R1/B1",  4:  "BM3R1/B2",  5:  "BM3R1/B3",
    6:  "HlyIIR/H1",
    7:  "IcaRA/I1",
    8:  "LitR/L1",
    9:  "LmrA/N1",
    10: "PhlF/P1",   11: "PhlF/P2",   12: "PhlF/P3",
    13: "PsrA/R1",
    14: "QacR/Q1",   15: "QacR/Q2",
    16: "SrpR/S1",   17: "SrpR/S2",   18: "SrpR/S3",   19: "SrpR/S4",
}

# Paper Fig. 3 yellow-dot designs (sim-rank-1 of each exemplar).
# slot_d2o = depth-to-output per regulator slot, computed once from the
# optimal_topology pickles in $DEEPCIRC_EXEMPLARS/0x{2B,17,6D}/.
PAPER_YELLOW_DOTS = [
    {"target": "0x2B", "size": 5, "circuit": 36.5869,  "growth": 0.7609,
     "perm": [2, 19, 15, 13, 12],
     "slot_d2o": [2, 1, 2, 2, 1]},
    {"target": "0x17", "size": 6, "circuit": 125.7974, "growth": 0.7590,
     "perm": [15, 11, 2, 13, 19, 1],
     "slot_d2o": [1, 2, 3, 2, 4, 4]},
    {"target": "0x6D", "size": 7, "circuit": 5.2859,   "growth": 0.6883,
     "perm": [1, 19, 13, 11, 0, 2, 3],
     "slot_d2o": [3, 2, 2, 1, 3, 2, 1]},
]

# Cluster overlay colours for the left-pane scatter.
CLUSTER_KNEE_LIGHT    = "#111111"   # near-black — 215 pareto-optimal dots
CLUSTER_MAXCIRC_LIGHT = MID_GRAY    # "#9E9E9E" — 215 max-circuit dots
CLUSTER_KNEE_DARK     = "#111111"   # pareto-optimal accent (legend / strip)
CLUSTER_MAXCIRC_DARK  = DARK_GRAY   # "#4A4A4A" — max-circuit accent

CANVAS_W_MM = 270
CANVAS_H_MM = 199


# --------------------------------------------------------------------- helpers


def _parse_source_list(src) -> list[str]:
    """Source column may be a JSON-like list ('["0x2B","0x4D"]') or a
    single token ('0x2B'). Always return list[str]."""
    s = str(src).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            return [s]
    return [s]


def _compute_slot_d2o(topo: dict) -> list[int]:
    """Return d2o (depth-to-output) per regulator slot, in slot-index
    order (ascending non-IO node id, matching gate_assignments)."""
    nodes = {n["id"]: n["type"] for n in topo["nodes"]}
    rev: dict = defaultdict(list)
    for u, v in topo["edges"]:
        rev[v].append(u)
    out_nodes = [nid for nid, t in nodes.items() if t == "OUT"]
    d2o: dict[int, int] = {}
    q = deque([(o, 0) for o in out_nodes])
    seen = set(out_nodes)
    while q:
        n, d = q.popleft()
        if n not in d2o or d < d2o[n]:
            d2o[n] = d
        for p in rev[n]:
            if p not in seen:
                seen.add(p); q.append((p, d + 1)); d2o[p] = d + 1
    regs = sorted([nid for nid, t in nodes.items() if t not in ("IN", "OUT")])
    return [d2o[r] for r in regs]


def _reorder_parts_by_position(parts: list[str], slot_d2o: list[int]
                                 ) -> list[str]:
    """Reorder a part list so input-proximal slots come first (high d2o
    $\\rightarrow$ low d2o). Stable secondary sort by original slot index keeps
    parts at the same d2o in their canonical node-id order."""
    indexed = list(enumerate(parts))
    indexed.sort(key=lambda ip: (-slot_d2o[ip[0]], ip[0]))
    return [p for _, p in indexed]


def _text_width_data(ax, text: str, fontsize: float,
                      fontweight: str = "normal") -> float:
    """Width of `text` in ax's data x-coords. Renders a temporary Text
    artist, measures its pixel bbox via the figure renderer, transforms
    back into data coords, removes the artist. Lets the part-name
    renderer position segments correctly in Arial (proportional)."""
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    t = ax.text(0, 0, text, fontsize=fontsize, fontweight=fontweight,
                va="center", ha="left")
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    p0 = ax.transData.inverted().transform((0, 0))
    p1 = ax.transData.inverted().transform((bbox.width, 0))
    return p1[0] - p0[0]


def _render_parts_cell(ax, x0: float, y: float, parts: list[str]) -> None:
    """Render the part list at (x0, y) with top-5 parts in coloured bold
    text (non-top-5 stay black). Each segment positioned via bbox
    measurement so it works in Arial (proportional widths)."""
    SEP = ", "
    sep_w = _text_width_data(ax, SEP, fontsize=7)
    x = x0
    for j, part in enumerate(parts):
        is_top5 = part in TOP5_COLORS
        color   = TOP5_COLORS[part] if is_top5 else "#000000"
        weight  = "bold" if is_top5 else "normal"
        ax.text(x, y, part,
                 fontsize=7, va="center", ha="left",
                 color=color, fontweight=weight)
        x += _text_width_data(ax, part, fontsize=7, fontweight=weight)
        if j < len(parts) - 1:
            ax.text(x, y, SEP,
                     fontsize=7, va="center", ha="left",
                     color="#000000")
            x += sep_w


def _render_top5_legend(ax, x0: float, y: float) -> None:
    """Compact legend strip. Part names are shown directly in their
    highlight colour — no swatch boxes."""
    PAD = 0.40
    label = "Top-5 parts:"
    ax.text(x0, y, label,
             fontsize=7, va="center", ha="left",
             color="#000000", fontweight="bold")
    x = x0 + _text_width_data(ax, label, fontsize=7, fontweight="bold") + PAD
    for part, color in TOP5_COLORS.items():
        ax.text(x, y, part,
                 fontsize=7, va="center", ha="left",
                 color=color, fontweight="bold")
        x += _text_width_data(ax, part, fontsize=7, fontweight="bold") + PAD


def _build_best_per_target(fronts: pd.DataFrame) -> pd.DataFrame:
    """Filter the full Pareto-front set by paper MLP thresholds, expand
    multi-source rows so each (design, target) pair contributes a row,
    then keep the single highest-circuit_log design per target."""
    fronts = fronts.copy()
    fronts["circuit_score"] = 2.0 ** fronts["circuit_log"]   # log_vec is log₂
    mask = ((fronts["circuit_score"] >= CIRC_SCORE_FLOOR)
            & (fronts["toxicity"] >= TOX_FLOOR))
    buildable = fronts[mask].copy()

    buildable["sources_list"] = buildable["source"].apply(_parse_source_list)

    rows: list[dict] = []
    for _, r in buildable.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)

    best = (long.sort_values("circuit_log", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = (best.sort_values("circuit_log", ascending=False)
                .reset_index(drop=True))
    best["rank"]         = best.index + 1
    best["npn_class"]    = best["target"].apply(S.npn_class_of)
    best["target_label"] = best["target"]
    return best


def _maxcirc_per_topology() -> pd.DataFrame:
    """One row per topology — its highest-circuit_log buildable design
    (paper MLP gates: circuit_score >= 2 AND growth >= 0.5)."""
    fronts = L.load_pareto_fronts("G3").copy()
    fronts["circuit_score"] = 2.0 ** fronts["circuit_log"]   # log_vec is log₂
    mask = ((fronts["circuit_score"] >= CIRC_SCORE_FLOOR)
            & (fronts["toxicity"] >= TOX_FLOOR))
    buildable = fronts[mask].copy()
    buildable = (buildable.sort_values("circuit_log", ascending=False)
                          .groupby("topology_id", as_index=False)
                          .first())
    buildable["npn_class"] = buildable["source"].apply(S.npn_class_of)
    buildable["target_label"] = buildable["source"].apply(S.parse_source)
    return buildable


def _top15_maxcirc_per_target() -> pd.DataFrame:
    """Top 15 max-circuit designs, 1 per target function, by circuit_log."""
    best = _build_best_per_target(L.load_pareto_fronts("G3"))
    best = (best.sort_values("circuit_log", ascending=False)
                 .head(15).copy())
    best["table_rank"]   = np.arange(1, len(best) + 1)
    best["target_label"] = best["target"]
    return best


def _top15_knee_per_target() -> pd.DataFrame:
    """Top 15 knees by combined_score at paper-MLP-gate floors, 1 per target.

    combined_score = (circuit_log − circ_min) / circ_range
                   + (toxicity   − tox_min)  / tox_range
    using PAPER MLP-gate floors (ln(2), 0.5) rather than Path-A floors,
    so we score the full visible cloud (not the stricter Path-A subset
    baked into cross_target_portfolio.csv).
    """
    k = L.load_pareto_knees("G3").copy()
    mask = ((k["knee_A_circuit_log"] >= CIRC_LOG_FLOOR)
            & (k["knee_A_toxicity"]   >= TOX_FLOOR))
    k = k[mask].copy()
    circ_range = float(k["knee_A_circuit_log"].max() - CIRC_LOG_FLOOR)
    tox_range  = 1.0 - TOX_FLOOR
    k["combined_score"] = (
        (k["knee_A_circuit_log"] - CIRC_LOG_FLOOR) / circ_range
        + (k["knee_A_toxicity"]  - TOX_FLOOR)      / tox_range
    )
    k["sources_list"] = k["source"].apply(_parse_source_list)
    rows = []
    for _, r in k.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)
    best = (long.sort_values("combined_score", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = (best.sort_values("combined_score", ascending=False)
                .head(15).copy())
    best["circuit_log"]   = best["knee_A_circuit_log"]
    best["toxicity"]      = best["knee_A_toxicity"]
    best["circuit_score"] = 2.0 ** best["circuit_log"]
    best["part_names"]    = best["knee_A_part_names"]
    best["target_label"]  = best["target"]
    best["npn_class"]     = best["target"].apply(S.npn_class_of)
    best["table_rank"]    = np.arange(1, len(best) + 1)
    return best


# --------------------------------------------------------------------- main


def build_panel() -> None:
    use_style()

    graphs = json.loads((REPO_ROOT / "data" / "topology_g3"
                         / "topology_graphs.json").read_text())
    topo_by_id = {t["topology_id"]: t for t in graphs["topologies"]}

    knees    = L.load_pareto_knees("G3")
    maxc_all = _maxcirc_per_topology()      # 215 per-topology max-circ
    knee15   = _top15_knee_per_target()      # 1-per-target, top 15
    maxc15   = _top15_maxcirc_per_target()   # 1-per-target, top 15

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM),
        gridspec_kw={"width_ratios": [1.50, 1.55]},
    )
    fig.subplots_adjust(
        # Pulled right edge in 0.985 $\\rightarrow$ 0.96 so the right table sits
        # back from the far figure edge; wspace 0.10 $\\rightarrow$ 0.04 to
        # tighten the gap between the left scatter and right table.
        left=0.05, right=0.96, top=0.93, bottom=0.05, wspace=0.04,
    )

    # =================================================================
    # LEFT pane — 215+215 cluster scatter + paper yellow-dots.
    # =================================================================
    ax1.scatter(
        knees["knee_A_circuit_log"], knees["knee_A_toxicity"],
        s=22, c=CLUSTER_KNEE_LIGHT, edgecolor="white",
        linewidth=0.4, alpha=0.90, zorder=4,
        label=f"Pareto-optimal - best joint score (n={len(knees)})",
    )
    ax1.scatter(
        maxc_all["circuit_log"], maxc_all["toxicity"],
        s=22, c=CLUSTER_MAXCIRC_LIGHT, edgecolor="white",
        linewidth=0.4, alpha=0.90, zorder=5,
        label=f"Max circuit score - highest circuit_log (n={len(maxc_all)})",
    )

    xlim_min = min(knees["knee_A_circuit_log"].min() - 0.3,
                   maxc_all["circuit_log"].min() - 0.3, -1.0)
    xlim_max = max(knees["knee_A_circuit_log"].max() + 0.4,
                   maxc_all["circuit_log"].max() + 0.4, 14.0)
    ylim_min = min(0.45, knees["knee_A_toxicity"].min() - 0.02,
                   maxc_all["toxicity"].min() - 0.02)
    ylim_max = max(knees["knee_A_toxicity"].max() + 0.02,
                   maxc_all["toxicity"].max() + 0.02, 1.0)
    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(ylim_min, ylim_max)

    rect = Rectangle(
        (CIRC_LOG_FLOOR, TOX_FLOOR),
        xlim_max - CIRC_LOG_FLOOR, ylim_max - TOX_FLOOR,
        facecolor=PASTEL["blue"], alpha=0.18,
        edgecolor=LIGHT_GRAY, linewidth=0.5, linestyle="--", zorder=1,
    )
    ax1.add_patch(rect)
    ax1.axvline(CIRC_LOG_FLOOR, color=LIGHT_GRAY, linestyle="--",
                 linewidth=0.5, alpha=1.0, zorder=2)
    ax1.axhline(TOX_FLOOR,  color=LIGHT_GRAY, linestyle="--",
                 linewidth=0.5, alpha=1.0, zorder=2)
    ax1.text(CIRC_LOG_FLOOR + 0.10, ylim_max - 0.005,
              "buildable (paper MLP gate)",
              fontsize=7, color="#000000", va="top", ha="left")

    # Paper Fig. 3 yellow-dot top picks rendered as plain filled
    # circles at the same s=22 size as the surrounding cluster dots
    # (matches the published convention). Per-target text labels
    # dropped 2026-05-29 — they live in the legend + the bottom YD
    # reference strip below the table.
    ax1.scatter(
        [float(np.log2(yd["circuit"])) for yd in PAPER_YELLOW_DOTS],
        [yd["growth"] for yd in PAPER_YELLOW_DOTS],
        marker="o", s=22,
        c=ACCENT_YELLOW, edgecolor=ACCENT_YELLOW_DARK,
        linewidth=0.6, zorder=8,
        label=f"Paper Fig. 3 yellow-dot (n={len(PAPER_YELLOW_DOTS)})",
    )

    ax1.set_xlabel(r"Circuit Score (log$_2$)",
                    fontsize=7, color="#000000", labelpad=2.5)
    ax1.set_ylabel("Growth Score (raw)",
                    fontsize=7, color="#000000", labelpad=2.5)
    # Per-axis title removed 2026-05-28 — replaced by a single panel-level
    # title rendered at the very end of build_panel().
    leg = ax1.legend(
        loc="upper left", frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fontsize=7,
        handletextpad=0.5, borderpad=0.4, labelspacing=0.35,
    )
    for txt in leg.get_texts():
        txt.set_color("#000000")
    ax1.grid(True, linewidth=0.25, alpha=1.0, color=LIGHT_GRAY, zorder=0)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax1.spines[spine].set_color("#000000")
        ax1.spines[spine].set_linewidth(0.6)
    ax1.tick_params(axis="both", which="major",
                     color=DARK_GRAY, width=0.6, length=2.5,
                     labelsize=7, labelcolor="#000000")

    # =================================================================
    # RIGHT pane — 30-row table + yellow-dot table + bottom legend.
    # =================================================================
    table_xlim = 16.0
    ax2.set_xlim(0, table_xlim)

    SECTION_HEADER_H = 0.6
    ROW_H            = 0.7
    bottom_legend_h  = 0.8
    YD_ROW_H         = 0.65
    YD_HEADER_H      = 0.55

    n_max  = len(maxc15)
    n_knee = len(knee15)

    y_col_header = 0.0
    y_sec_a_hdr  = 0.40
    y_sec_a_top  = y_sec_a_hdr + SECTION_HEADER_H
    y_sec_a_bot  = y_sec_a_top + n_max * ROW_H
    y_sec_b_hdr  = y_sec_a_bot + 0.20
    y_sec_b_top  = y_sec_b_hdr + SECTION_HEADER_H
    y_sec_b_bot  = y_sec_b_top + n_knee * ROW_H
    y_yd_label   = y_sec_b_bot + 0.45
    # YD column headers removed — collapse the old (header gap +
    # header height) into a single ~0.50-unit gap from the section
    # label to the first data row so the trio of YD entries sits
    # close under the title.
    y_yd_top     = y_yd_label  + 0.50
    y_yd_bot     = y_yd_top    + len(PAPER_YELLOW_DOTS) * YD_ROW_H
    y_legend     = y_yd_bot    + 0.45

    # Top-of-table whitespace tightened (was -1.0) so the column
    # header row sits closer to the top of the axes box, aligning
    # with the top of the buildable blue shaded region in ax1.
    ax2.set_ylim(-0.3, y_legend + bottom_legend_h + 0.3)
    ax2.invert_yaxis()
    ax2.axis("off")

    # Column header bar — abbreviated c_log/c_score to fit the column
    # slots without overrunning into adjacent columns.
    cols_x  = [0.1, 1.0, 2.3, 3.4, 4.9, 6.4, 7.8]
    headers = ["Rank", "Target", "Size", "Circuit (l)", "Circuit (r)", "Growth",
               "Part Names (input $\\rightarrow$ output, position-informed)"]
    for x, h in zip(cols_x, headers):
        ax2.text(x, y_col_header, h,
                  fontsize=7, fontweight="normal", color="#111",
                  va="center", ha="left")
    ax2.plot([cols_x[0] - 0.05, table_xlim - 0.05],
              [y_col_header + 0.30, y_col_header + 0.30],
              color="#000000", linewidth=0.6)

    def render_section(df, y_hdr_top, y_first_row, section_title, *,
                        accent_color):
        ax2.add_patch(Rectangle(
            (cols_x[0] - 0.05, y_hdr_top),
            table_xlim - cols_x[0], SECTION_HEADER_H * 0.9,
            facecolor=PASTEL["blue"], edgecolor="none", zorder=0,
        ))
        ax2.add_patch(Rectangle(
            (cols_x[0] - 0.05, y_hdr_top),
            0.12, SECTION_HEADER_H * 0.9,
            facecolor=accent_color, edgecolor="none", zorder=1,
        ))
        ax2.text(cols_x[0] + 0.12, y_hdr_top + SECTION_HEADER_H * 0.45,
                  section_title,
                  fontsize=7, fontweight="normal", color="#111",
                  va="center", ha="left")
        for i, (_, r) in enumerate(df.iterrows()):
            y = y_first_row + i * ROW_H + ROW_H * 0.5 - 0.05
            ax2.text(cols_x[0], y, f"#{int(r['table_rank'])}",
                      fontsize=7, va="center",
                      fontweight="normal", color="#000000")
            # Per-target colour chip removed 2026-05-28 — hex label alone.
            ax2.text(cols_x[1], y, r["target_label"],
                      fontsize=7, va="center", color="#000000")
            ax2.text(cols_x[2], y, f"{int(r['gate_count'])}-reg",
                      fontsize=7, va="center", color="#000000")
            ax2.text(cols_x[3], y, f"{r['circuit_log']:.2f}",
                      fontsize=7, va="center",
                      color="#000000")
            ax2.text(cols_x[4], y, f"{r['circuit_score']:.1f}",
                      fontsize=7, va="center",
                      color="#000000")
            ax2.text(cols_x[5], y, f"{r['toxicity']:.3f}",
                      fontsize=7, va="center",
                      color="#000000")
            parts = (ast.literal_eval(r["part_names"])
                     if isinstance(r["part_names"], str)
                     else list(r["part_names"]))
            slot_d2o = _compute_slot_d2o(topo_by_id[r["topology_id"]])
            parts = _reorder_parts_by_position(parts, slot_d2o)
            _render_parts_cell(ax2, cols_x[6], y, parts)

    # Section headers — plain black, non-bold; new "pareto-optimal"
    # terminology replaces "knee".
    render_section(
        maxc15, y_sec_a_hdr, y_sec_a_top,
        "Top 15 By Max Circuit Score: best per target by circuit score",
        accent_color=CLUSTER_MAXCIRC_DARK,
    )
    render_section(
        knee15, y_sec_b_hdr, y_sec_b_top,
        "Top 15 by Pareto-Optimal: best per target by combined score "
        "(normalized circuit score × growth score)",
        accent_color=CLUSTER_KNEE_DARK,
    )

    ax2.plot([cols_x[0] - 0.05, table_xlim - 0.05],
              [y_sec_b_bot + 0.05, y_sec_b_bot + 0.05],
              color=DARK_GRAY, linewidth=0.5)

    # ---- Yellow-dot reference table (paper Fig. 3 sim-rank-1 designs)
    # Label de-yellowed + black (non-bold) per user direction.
    ax2.text(cols_x[0], y_yd_label,
              "Paper Fig. 3 yellow-dots  "
              "(scores below are simulator ground truth, NOT MLP predictions):",
              fontsize=7, fontweight="normal", va="center",
              ha="left", color="#111")

    # YD column headers REMOVED — main column header row above (rank /
    # target / size / c_log / c_score / growth / part_names ...) already
    # labels these columns; repeating them in the YD strip was redundant.
    ax2.plot([cols_x[0] - 0.05, table_xlim - 0.05],
              [y_yd_top - 0.02, y_yd_top - 0.02],
              color=MID_GRAY, linewidth=0.4)

    for i, yd in enumerate(PAPER_YELLOW_DOTS):
        y = y_yd_top + i * YD_ROW_H + YD_ROW_H * 0.5
        yd_circ_log = float(np.log2(yd["circuit"]))
        # Plain filled circle (paper Fig 3 convention).
        ax2.scatter([cols_x[0] + 0.10], [y],
                     marker="o", s=28,
                     c=ACCENT_YELLOW, edgecolor=ACCENT_YELLOW_DARK,
                     linewidth=0.6, zorder=4, clip_on=False)
        ax2.text(cols_x[1] + 0.15, y, yd["target"],
                  fontsize=7, va="center", color="#000000")
        ax2.text(cols_x[2], y, f"{yd['size']}-reg",
                  fontsize=7, va="center", color="#000000")
        ax2.text(cols_x[3], y, f"{yd_circ_log:.2f}",
                  fontsize=7, va="center",
                  color="#000000")
        ax2.text(cols_x[4], y, f"{yd['circuit']:.1f}",
                  fontsize=7, va="center",
                  color="#000000")
        ax2.text(cols_x[5], y, f"{yd['growth']:.3f}",
                  fontsize=7, va="center",
                  color="#000000")
        yd_parts = [LIBRARY_PART_NAMES[i] for i in yd["perm"]]
        yd_parts = _reorder_parts_by_position(yd_parts, yd["slot_d2o"])
        _render_parts_cell(ax2, cols_x[6], y, yd_parts)

    _render_top5_legend(ax2, x0=cols_x[0], y=y_legend + 0.3)

    # Panel-level title — replaces the old "Best knee design vs max-circuit
    # design ..." per-axis title; spans the full panel above both panes.
    fig.text(0.5, 0.975,
              "Pareto-optimal vs max circuit score design space "
              "and part composition",
              ha="center", va="top",
              fontsize=12, fontweight="normal", color="#000000")

    out_base = PANELS_VEC / "panel_c"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel C written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
