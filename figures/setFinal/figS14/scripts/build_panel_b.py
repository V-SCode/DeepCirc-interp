"""Figure S05 — Panel B: per-part circuit log-odds beeswarm.

Circuit-score counterpart of Panel A. Same beeswarm layout, but the
task is `circuit_log` (not `toxicity`), so this panel surfaces which
parts the agent places in the top-5% / bot-5% of designs by circuit
margin rather than by growth. Together with Panel A they form a paired
view of the composition-to-function map: A = growth, B = circuit.
(Predecessor: the original Panel B was the QacR/Q1-vs-Q2 paired
role-bar comparison ported from `fig02_toxic_class`; preserved at
`scripts/_archive/build_panel_b_qacr_pair_bars.py`.)

Layout:
  y-axis = 20 library parts sorted by median log-odds (most-destructive
           at the bottom, most-constructive at the top).
  x-axis = log-odds(top-5% / bot-5% by circuit_log) per (part, fp_key)
           cell that passes the universal-eligible filter
           (n_topologies >= 5, n_targets >= 3, total_designs >= 1000).
  per-part dots = one per fp_key cell, jittered vertically.
  vertical reference lines at log_odds = 0, -2 (destructive thr),
                                          +1 (constructive thr).
  horizontal banding by class region (destructive at bottom,
  constructive at top).

Class semantics (driven by per-part median log_odds at 5/5):
  destructive_ex  $\\rightarrow$ PASTEL["red"]    (PhlF/P1 — median -1.47, sole outlier)
  destructive_t2  $\\rightarrow$ PASTEL["orange"] (SrpR/S1, HlyIIR/H1, LitR/L1, LmrA/N1
                                       — medians -1.20 to -0.87)
  constructive    $\\rightarrow$ PASTEL["green"]  (PhlF/P2, PhlF/P3, SrpR/S2, SrpR/S4,
                                       AmtR/A1 — medians +0.40 to +0.56)
  neutral         $\\rightarrow$ LIGHT_GRAY       (most parts)

Source: data/topology_g3/l2_top05/l2_enrichment.csv

Outputs:
  panels/vector/panel_b.pdf | .svg
  panels/raster/panel_b.png

Run:
    python \\
        figures/setFinal/figS14/scripts/build_panel_b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

L2_TOP05_CSV = (REPO_ROOT / "data" / "topology_g3"
                / "l2_top05" / "l2_enrichment.csv")

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    PASTEL, DARK_GRAY, MID_GRAY, LIGHT_GRAY,
    ACCENT_BLUE,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS14"
              / "panels" / "vector")

TEXT_GREY = "#666666"

PART_CLASS_COLORS = {
    "destructive_ex":  "#111111",   # near-black — strongest "destructive" emphasis
    "destructive_t2":  MID_GRAY,    # "#9E9E9E" — second-tier destructive
    "constructive":    ACCENT_BLUE,
    "neutral":         LIGHT_GRAY,
}

PART_HIGHLIGHTS = {
    "PhlF/P1":    ("destructive_ex", "destructive (extreme)"),
    "SrpR/S1":    ("destructive_t2", "destructive"),
    "HlyIIR/H1":  ("destructive_t2", "destructive"),
    "LitR/L1":    ("destructive_t2", "destructive"),
    "LmrA/N1":    ("destructive_t2", "destructive"),
    "PhlF/P2":    ("constructive",   "constructive"),
    "PhlF/P3":    ("constructive",   "constructive"),
    "SrpR/S2":    ("constructive",   "constructive"),
    "SrpR/S4":    ("constructive",   "constructive"),
    "AmtR/A1":    ("constructive",   "constructive"),
}


CANVAS_W_MM = 200
CANVAS_H_MM = 165


def build_panel() -> None:
    use_style()

    l2 = pd.read_csv(L2_TOP05_CSV)
    l2_eligible = l2[
        (l2["task"] == "circuit_log")
        & (l2["n_topologies"] >= 5)
        & (l2["n_targets"] >= 3)
        & (l2["total_designs"] >= 1000)
    ].copy()

    parts_by_median = (
        l2_eligible.groupby("part_name")["log_odds"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes((0.21, 0.08, 0.74, 0.78))

    # --- class-region banding (drawn first, behind everything) ----------
    destructive_idxs = [i for i, p in enumerate(parts_by_median)
                        if PART_HIGHLIGHTS.get(p, ("neutral",))[0] in
                        ("destructive_ex", "destructive_t2")]
    constructive_idxs = [i for i, p in enumerate(parts_by_median)
                         if PART_HIGHLIGHTS.get(p, ("neutral",))[0]
                         == "constructive"]
    if destructive_idxs:
        # Destructive-region background tint swapped from pastel red $\\rightarrow$
        # DARK_GRAY to align with the paper palette; per-part dots,
        # ticks, threshold markers + legend keep their red/orange
        # identity (per user direction).
        ax.axhspan(min(destructive_idxs) - 0.5, max(destructive_idxs) + 0.5,
                    color=DARK_GRAY, alpha=0.10, zorder=0)
    if constructive_idxs:
        ax.axhspan(min(constructive_idxs) - 0.5, max(constructive_idxs) + 0.5,
                    color=PART_CLASS_COLORS["constructive"], alpha=0.12,
                    zorder=0)

    # --- per-part jittered dots + median tick ---------------------------
    rng = np.random.default_rng(42)
    for i, part in enumerate(parts_by_median):
        cells = l2_eligible[l2_eligible["part_name"] == part]
        x = cells["log_odds"].values
        y = i + rng.uniform(-0.30, 0.30, size=len(x))

        category = PART_HIGHLIGHTS.get(part, ("neutral", "other"))[0]
        color = PART_CLASS_COLORS[category]
        edge = DARK_GRAY if category != "neutral" else "none"
        zorder = 5 if category != "neutral" else 2
        size = 9 if category != "neutral" else 6
        alpha = 0.95 if category != "neutral" else 0.55

        ax.scatter(x, y, s=size, c=color, edgecolor=edge, linewidth=0.3,
                    alpha=alpha, zorder=zorder)

        med = np.median(x)
        ax.plot([med, med], [i - 0.40, i + 0.40],
                 color=DARK_GRAY, linewidth=0.9, zorder=6)

    # --- reference lines ------------------------------------------------
    # Thresholds in log₂ space (rescaled 2026-06-01 to preserve the
    # semantic boundaries used pre-log₂-migration: destructive ≈ −3,
    # constructive ≈ +1.5).
    ax.axvline(0, color=MID_GRAY, linestyle=":", linewidth=0.6, zorder=1)
    ax.axvline(-3, color=PART_CLASS_COLORS["destructive_t2"], linestyle="--",
                linewidth=0.55, alpha=0.7, zorder=1)
    ax.axvline(1.5, color=PART_CLASS_COLORS["constructive"], linestyle="--",
                linewidth=0.55, alpha=0.7, zorder=1)

    # --- y-axis labels --------------------------------------------------
    ax.set_yticks(range(len(parts_by_median)))
    ax.set_yticklabels(parts_by_median, fontsize=12)
    for tick, part in zip(ax.get_yticklabels(), parts_by_median):
        if part in PART_HIGHLIGHTS:
            tick.set_color(PART_CLASS_COLORS[PART_HIGHLIGHTS[part][0]])
        else:
            tick.set_color("#000000")

    ax.set_xlabel(r"log$_2$ enrichment (top-5% / bot-5% by circuit score (log))",
                   fontsize=12, color="#000000", labelpad=2.0)
    ax.set_xlim(left=l2_eligible["log_odds"].min() - 0.4,
                 right=l2_eligible["log_odds"].max() + 0.4)
    ax.set_ylim(-0.6, len(parts_by_median) - 0.4)
    ax.tick_params(axis="x", labelsize=12, labelcolor="#000000",
                    color="#000000", width=0.6, length=2.5)
    ax.tick_params(axis="y", color="#000000", width=0.6, length=2.5)

    # --- threshold labels ON the dashed lines mid-plot ------------------
    mid_y = len(parts_by_median) / 2
    ax.text(-3, mid_y, "destructive thr",
             color=PART_CLASS_COLORS["destructive_t2"],
             fontsize=12, ha="right", va="center",
             rotation=90, fontweight="normal",
             bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                        edgecolor="none", alpha=0.85), zorder=7)
    ax.text(1.5, mid_y, "constructive thr",
             color=PART_CLASS_COLORS["constructive"],
             fontsize=12, ha="left", va="center",
             rotation=90, fontweight="normal",
             bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                        edgecolor="none", alpha=0.85), zorder=7)

    # --- title ----------------------------------------------------------
    # Rendered via fig.text() (matches figS05 Panel D / E convention).
    # x=0.58 lines the title up with the axes centre (axes is at
    # left=0.21 width=0.74 $\\rightarrow$ centre 0.58); using fig-x=0.5 would
    # shift the title ~8% left of the plot content below.
    fig.text(0.58, 0.95, "Per part circuit score probability enrichment",
              ha="center", va="top",
              fontsize=19, fontweight="normal", color="#000000")

    # --- legend ---------------------------------------------------------
    handles = [
        mpatches.Patch(color=PART_CLASS_COLORS["destructive_ex"],
                        label="destructive extreme"),
        mpatches.Patch(color=PART_CLASS_COLORS["destructive_t2"],
                        label="destructive"),
        mpatches.Patch(color=PART_CLASS_COLORS["constructive"],
                        label="constructive"),
        mpatches.Patch(color=PART_CLASS_COLORS["neutral"],
                        label="neutral"),
    ]
    # labelspacing 0.25 $\\rightarrow$ 0.5 + handlelength/handleheight = 1.0 so the
    # swatches render as rough squares (default handlelength=2.0 made
    # them ~3× wider than tall = horizontally stretched) and the rows
    # have normal vertical breathing room (was visibly squished).
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                     fontsize=12, handletextpad=0.4,
                     handlelength=1.0, handleheight=1.0,
                     labelspacing=0.5, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color("#000000")

    # --- paper axis styling --------------------------------------------
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(0.6)

    out_base = PANELS_VEC / "panel_b"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel B written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
