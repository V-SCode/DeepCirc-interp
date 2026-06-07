"""figS04 Panel C (draft variant d) — knee designs vs max-circuit designs.

Goal: surface BOTH design-space distributions across all 215 topologies
simultaneously, so we can compare where the circuit-only oracle sends
its picks vs where the joint-Pareto oracle sends them.

Left pane (scatter):
  * 215 LIGHT BLUE dots   — each topology's best Pareto KNEE design
                            (knee_A_circuit_log, knee_A_toxicity).
  * 215 LIGHT PURPLE dots — each topology's MAX-CIRCUIT design
                            (highest circuit_log among that topology's
                            buildable Pareto-front designs at the paper
                            MLP gates: circuit_score >= 2 AND
                            growth >= 0.5).
  No NPN colouring, no yellow-dot stars — this draft is about the
  knee-vs-maxcirc contrast only.

Right pane (table, 30 rows = 15 + 15):
  * TOP 15 BY MAX-CIRCUIT  — top 15 of the 215 max-circuit-per-topology
                              designs, sorted by circuit_log desc.
  * TOP 15 BY KNEE          — top 15 of cross_target_portfolio.csv
                              (24 entries, ranked by combined_score
                              = Plan-1 perpendicular-distance-from-chord).
  No top title, no top legend, no paper yellow-dot strip. Top-5 parts
  legend lives at the bottom.

Outputs:
  panels/vector/panel_draft_c_knee_vs_maxcirc.{pdf,svg}
  panels/raster/panel_draft_c_knee_vs_maxcirc.png
"""
from __future__ import annotations

import ast
import json
import sys
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

sys.path.insert(0, str(SCRIPT_PATH.parent))
import build_panel_c as bc  # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    DARK_GRAY, MID_GRAY, LIGHT_GRAY, PASTEL,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS13"
              / "panels" / "vector")

TEXT_GREY = "#666666"

# Light cluster colours for the 215+215 scatter, distinct from top-5
# part palette.
CLUSTER_KNEE_LIGHT    = PASTEL["blue"]    # "#9EC7E8" — 215 knees
CLUSTER_MAXCIRC_LIGHT = PASTEL["purple"]  # "#C5A1D5" — 215 max-circuit
# Saturated companions for the section-header accents in the table.
CLUSTER_KNEE_DARK     = "#1565C0"
CLUSTER_MAXCIRC_DARK  = "#7E57C2"

CIRC_SCORE_FLOOR = bc.CIRC_SCORE_FLOOR
CIRC_LOG_FLOOR   = bc.CIRC_LOG_FLOOR
TOX_FLOOR        = bc.TOX_FLOOR

TOP5_COLORS        = bc.TOP5_COLORS
LIBRARY_PART_NAMES = bc.LIBRARY_PART_NAMES

CANVAS_W_MM = 280
CANVAS_H_MM = 200


# --------------------------------------------------------------------- helpers

def _maxcirc_per_topology() -> pd.DataFrame:
    """One row per topology — its highest-circuit_log buildable design.

    Buildable = paper MLP gates (circuit_score >= 2 AND growth >= 0.5).
    Same source as figS04 Panel C canonical (all_topology_fronts.csv.gz)."""
    fronts = L.load_pareto_fronts("G3").copy()
    fronts["circuit_score"] = np.exp(fronts["circuit_log"])
    mask = ((fronts["circuit_score"] >= CIRC_SCORE_FLOOR)
            & (fronts["toxicity"] >= TOX_FLOOR))
    buildable = fronts[mask].copy()
    # Highest circuit_log row per topology.
    buildable = (buildable.sort_values("circuit_log", ascending=False)
                          .groupby("topology_id", as_index=False)
                          .first())
    buildable["npn_class"] = buildable["source"].apply(S.npn_class_of)
    # target_label keeps the first source string for multi-source rows
    # — it's just for the table; the scatter doesn't need it.
    buildable["target_label"] = buildable["source"].apply(S.parse_source)
    return buildable


def _top15_maxcirc_per_target() -> pd.DataFrame:
    """Top 15 max-circuit designs deduplicated to 1 per target function.

    Pulls the per-target best (n=20) from bc._build_best_per_target and
    takes the top 15 by circuit_log. The scatter still shows the
    per-topology 215-point cloud; the TABLE deliberately uses per-target
    diversity so we see what the oracle picks for each Boolean function.
    """
    best = bc._build_best_per_target(L.load_pareto_fronts("G3"))
    best = (best.sort_values("circuit_log", ascending=False)
                 .head(15).copy())
    best["table_rank"] = np.arange(1, len(best) + 1)
    best["target_label"] = best["target"]
    return best


def _top15_maxcirc(maxc_all: pd.DataFrame) -> pd.DataFrame:
    """(legacy) Top 15 of the per-topology max-circuit set, ranked by
    circuit_log. Kept around in case we want the per-topology view back."""
    top = (maxc_all.sort_values("circuit_log", ascending=False)
                   .head(15).copy())
    top["table_rank"] = np.arange(1, len(top) + 1)
    return top


def _top15_knee_per_target() -> pd.DataFrame:
    """Top 15 knees by RECOMPUTED combined_score at paper-MLP-gate floors,
    deduplicated to 1 design per target function.

    Method:
      1. Filter all 215 knees by paper-MLP-gate (circuit_log ≥ ln(2)
         AND toxicity ≥ 0.5).
      2. Compute combined_score with PAPER-MLP-gate floors using the
         Plan-1 formula:
             combined_score = (circuit_log − circ_min) / circ_range
                            + (toxicity   − tox_min)  / tox_range
         (Same formula as topology/scripts/14_l1_pareto_frontier
         .py L523-527 but at the looser MLP-gate floors so we score the
         full visible cloud, not just the Path-A subset.)
      3. Expand multi-source rows so each (knee_design, target_function)
         pair contributes a candidate row.
      4. For each target function, keep the single highest-combined_score
         row.
      5. Sort by combined_score desc, take top 15.

    Result: up to 15 distinct target functions in the table, each
    represented by its single best-joint-quality knee design.
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

    # Expand multi-source rows so each (knee, target) gets a candidate.
    k["sources_list"] = k["source"].apply(bc._parse_source_list)
    rows = []
    for _, r in k.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)

    # Per-target highest combined_score.
    best = (long.sort_values("combined_score", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = (best.sort_values("combined_score", ascending=False)
                .head(15).copy())
    best["circuit_log"]   = best["knee_A_circuit_log"]
    best["toxicity"]      = best["knee_A_toxicity"]
    best["circuit_score"] = np.exp(best["circuit_log"])
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
    maxc_all = _maxcirc_per_topology()        # 215 per-topology max-circ (scatter)
    knee15   = _top15_knee_per_target()        # 1-per-target, top 15
    maxc15   = _top15_maxcirc_per_target()     # 1-per-target, top 15

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM),
        gridspec_kw={"width_ratios": [1.50, 1.55]},
    )
    fig.subplots_adjust(
        left=0.05, right=0.985, top=0.96, bottom=0.05, wspace=0.10,
    )

    # =================================================================
    # LEFT pane — TWO 215-point clusters showing the full design-space
    # contrast. No NPN colouring, no yellow stars.
    # =================================================================
    # KNEE cluster (light blue) — 215 best Pareto knees, one per topology.
    ax1.scatter(
        knees["knee_A_circuit_log"], knees["knee_A_toxicity"],
        s=22, c=CLUSTER_KNEE_LIGHT, edgecolor="white",
        linewidth=0.4, alpha=0.90, zorder=4,
        label=f"KNEE — best joint Pareto (n={len(knees)})",
    )
    # MAX-CIRCUIT cluster (light purple) — 215 max-circuit per topology.
    ax1.scatter(
        maxc_all["circuit_log"], maxc_all["toxicity"],
        s=22, c=CLUSTER_MAXCIRC_LIGHT, edgecolor="white",
        linewidth=0.4, alpha=0.90, zorder=5,
        label=f"MAX-CIRCUIT — highest circuit_log (n={len(maxc_all)})",
    )

    xlim_min = min(knees["knee_A_circuit_log"].min() - 0.3,
                   maxc_all["circuit_log"].min() - 0.3, -0.5)
    xlim_max = max(knees["knee_A_circuit_log"].max() + 0.4,
                   maxc_all["circuit_log"].max() + 0.4, 9.5)
    ylim_min = min(0.45, knees["knee_A_toxicity"].min() - 0.02,
                   maxc_all["toxicity"].min() - 0.02)
    ylim_max = max(knees["knee_A_toxicity"].max() + 0.02,
                   maxc_all["toxicity"].max() + 0.02, 1.0)
    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(ylim_min, ylim_max)

    # Buildable region at paper MLP gates.
    rect = Rectangle(
        (CIRC_LOG_FLOOR, TOX_FLOOR),
        xlim_max - CIRC_LOG_FLOOR, ylim_max - TOX_FLOOR,
        facecolor=PASTEL["green"], alpha=0.18,
        edgecolor=LIGHT_GRAY, linewidth=0.5, linestyle="--", zorder=1,
    )
    ax1.add_patch(rect)
    ax1.axvline(CIRC_LOG_FLOOR, color=LIGHT_GRAY, linestyle="--",
                 linewidth=0.5, alpha=1.0, zorder=2)
    ax1.axhline(TOX_FLOOR,  color=LIGHT_GRAY, linestyle="--",
                 linewidth=0.5, alpha=1.0, zorder=2)
    ax1.text(CIRC_LOG_FLOOR + 0.10, ylim_max - 0.005,
              "buildable (paper MLP gate)",
              fontsize=6.0, color=TEXT_GREY, va="top", ha="left")

    ax1.set_xlabel(
        "circuit_log  (higher = better logic margin)",
        fontsize=6.0, color=TEXT_GREY, labelpad=2.5,
    )
    ax1.set_ylabel("growth score  (higher = healthier cell)",
                    fontsize=6.0, color=TEXT_GREY, labelpad=2.5)
    ax1.set_title(
        "Best knee design vs max-circuit design — one dot per topology, "
        "two clusters",
        loc="left", pad=6,
        fontsize=6.5, color=TEXT_GREY,
    )
    leg = ax1.legend(
        loc="upper left", frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fontsize=5.8,
        handletextpad=0.5, borderpad=0.4, labelspacing=0.35,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_GREY)
    ax1.grid(True, linewidth=0.25, alpha=1.0, color=LIGHT_GRAY, zorder=0)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax1.spines[spine].set_color(DARK_GRAY)
        ax1.spines[spine].set_linewidth(0.6)
    ax1.tick_params(axis="both", which="major",
                     color=DARK_GRAY, width=0.6, length=2.5,
                     labelsize=5.5, labelcolor=DARK_GRAY)

    # =================================================================
    # RIGHT pane — 30-row table (top-15 of each cluster) + bottom legend.
    # No top title, no top legend, no yellow-dot strip.
    # =================================================================
    table_xlim = 16.0
    ax2.set_xlim(0, table_xlim)

    SECTION_HEADER_H = 0.6
    ROW_H            = 0.7
    bottom_legend_h  = 0.8

    n_max  = len(maxc15)
    n_knee = len(knee15)

    y_col_header = 0.0
    y_sec_a_hdr  = 0.55
    y_sec_a_top  = y_sec_a_hdr + SECTION_HEADER_H
    y_sec_a_bot  = y_sec_a_top + n_max * ROW_H
    y_sec_b_hdr  = y_sec_a_bot + 0.25
    y_sec_b_top  = y_sec_b_hdr + SECTION_HEADER_H
    y_sec_b_bot  = y_sec_b_top + n_knee * ROW_H
    y_legend     = y_sec_b_bot + 0.6

    ax2.set_ylim(-1.0, y_legend + bottom_legend_h + 0.3)
    ax2.invert_yaxis()
    ax2.axis("off")

    cols_x  = [0.1, 1.1, 2.2, 3.2, 4.6, 5.9, 7.1]
    headers = ["rank", "target", "size", "circuit_log",
               "circuit_score", "growth",
               "part_names (input → output, position-informed)"]
    for x, h in zip(cols_x, headers):
        ax2.text(x, y_col_header, h,
                  fontsize=5.5, fontweight="bold", color="#111",
                  va="center", ha="left")
    ax2.plot([cols_x[0] - 0.05, table_xlim - 0.05],
              [y_col_header + 0.30, y_col_header + 0.30],
              color=DARK_GRAY, linewidth=0.6)

    def render_section(df, y_hdr_top, y_first_row, section_title, *,
                        accent_color):
        ax2.add_patch(Rectangle(
            (cols_x[0] - 0.05, y_hdr_top),
            table_xlim - cols_x[0], SECTION_HEADER_H * 0.9,
            facecolor="#F4F4F4", edgecolor="none", zorder=0,
        ))
        ax2.add_patch(Rectangle(
            (cols_x[0] - 0.05, y_hdr_top),
            0.12, SECTION_HEADER_H * 0.9,
            facecolor=accent_color, edgecolor="none", zorder=1,
        ))
        ax2.text(cols_x[0] + 0.12, y_hdr_top + SECTION_HEADER_H * 0.45,
                  section_title,
                  fontsize=6.0, fontweight="bold", color="#222",
                  va="center", ha="left")
        for i, (_, r) in enumerate(df.iterrows()):
            y = y_first_row + i * ROW_H + ROW_H * 0.5 - 0.05
            npn_color = S.NPN_CLASS_COLORS.get(r["npn_class"], "#999999")
            ax2.text(cols_x[0], y, f"#{int(r['table_rank'])}",
                      fontsize=5.0, va="center",
                      fontweight="normal", color=TEXT_GREY)
            ax2.add_patch(Rectangle(
                (cols_x[1] - 0.08, y - 0.12), 0.16, 0.24,
                facecolor=npn_color, edgecolor="none",
            ))
            ax2.text(cols_x[1] + 0.15, y, r["target_label"],
                      fontsize=5.0, va="center", color=TEXT_GREY)
            ax2.text(cols_x[2], y, f"{int(r['gate_count'])}-reg",
                      fontsize=5.0, va="center", color=TEXT_GREY)
            ax2.text(cols_x[3], y, f"{r['circuit_log']:.2f}",
                      fontsize=5.0, va="center", family="monospace",
                      color=TEXT_GREY)
            ax2.text(cols_x[4], y, f"{r['circuit_score']:.1f}",
                      fontsize=5.0, va="center", family="monospace",
                      color=TEXT_GREY)
            ax2.text(cols_x[5], y, f"{r['toxicity']:.3f}",
                      fontsize=5.0, va="center", family="monospace",
                      color=TEXT_GREY)
            parts = (ast.literal_eval(r["part_names"])
                     if isinstance(r["part_names"], str)
                     else list(r["part_names"]))
            slot_d2o = bc._compute_slot_d2o(topo_by_id[r["topology_id"]])
            parts = bc._reorder_parts_by_position(parts, slot_d2o)
            bc._render_parts_cell(ax2, cols_x[6], y, parts)

    render_section(
        maxc15, y_sec_a_hdr, y_sec_a_top,
        "TOP 15 BY MAX-CIRCUIT  (best per target function by circuit_log, "
        "paper MLP gates)",
        accent_color=CLUSTER_MAXCIRC_DARK,
    )
    render_section(
        knee15, y_sec_b_hdr, y_sec_b_top,
        "TOP 15 BY KNEE  (best per target function by combined_score; "
        "normalised circuit_log + growth above paper MLP gates)",
        accent_color=CLUSTER_KNEE_DARK,
    )

    ax2.plot([cols_x[0] - 0.05, table_xlim - 0.05],
              [y_sec_b_bot + 0.05, y_sec_b_bot + 0.05],
              color=DARK_GRAY, linewidth=0.5)

    bc._render_top5_legend(ax2, x0=cols_x[0], y=y_legend + 0.3)

    out_base = PANELS_VEC / "panel_draft_c_knee_vs_maxcirc"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("panel_draft_c_knee_vs_maxcirc written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
