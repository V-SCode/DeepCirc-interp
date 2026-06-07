"""Figure S05 — Panel B v2: blue→red palette variant.

Same data + layout as build_panel_b.py; "bad" tiers (destructive_ex,
destructive_t2) swapped from near-black / MID_GRAY to deep-red /
tier-2-red. Background destructive-region tint also red (low alpha).
Compare against canonical panel_b.pdf for blue-to-red vs blue-to-grey.

Outputs: panels/vector/panel_b_v2.{pdf,svg}
         panels/raster/panel_b_v2.png

Run:
    python \\
        figures/setFinal/figS14/scripts/build_panel_b_v2.py
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
    DARK_GRAY, MID_GRAY, LIGHT_GRAY, ACCENT_BLUE,
)

sys.path.insert(0, str(SCRIPT_PATH.parent))
from _palette_v2 import BAD_DEEP, BAD_TIER2, BAD_BAND  # noqa: E402


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS14"
              / "panels" / "vector")

TEXT_GREY = "#666666"

PART_CLASS_COLORS = {
    "destructive_ex":  BAD_DEEP,    # deep red — strongest "destructive"
    "destructive_t2":  BAD_TIER2,   # tier-2 red
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


CANVAS_W_MM = 130
CANVAS_H_MM = 120


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
    ax = fig.add_axes((0.21, 0.10, 0.74, 0.84))

    destructive_idxs = [i for i, p in enumerate(parts_by_median)
                        if PART_HIGHLIGHTS.get(p, ("neutral",))[0] in
                        ("destructive_ex", "destructive_t2")]
    constructive_idxs = [i for i, p in enumerate(parts_by_median)
                         if PART_HIGHLIGHTS.get(p, ("neutral",))[0]
                         == "constructive"]
    if destructive_idxs:
        ax.axhspan(min(destructive_idxs) - 0.5, max(destructive_idxs) + 0.5,
                    color=BAD_BAND, alpha=0.08, zorder=0)
    if constructive_idxs:
        ax.axhspan(min(constructive_idxs) - 0.5, max(constructive_idxs) + 0.5,
                    color=PART_CLASS_COLORS["constructive"], alpha=0.12,
                    zorder=0)

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

    # Thresholds in log₂ space (rescaled 2026-06-01).
    ax.axvline(0, color=MID_GRAY, linestyle=":", linewidth=0.6, zorder=1)
    ax.axvline(-3, color=PART_CLASS_COLORS["destructive_t2"], linestyle="--",
                linewidth=0.55, alpha=0.8, zorder=1)
    ax.axvline(1.5, color=PART_CLASS_COLORS["constructive"], linestyle="--",
                linewidth=0.55, alpha=0.7, zorder=1)

    ax.set_yticks(range(len(parts_by_median)))
    ax.set_yticklabels(parts_by_median, fontsize=5.5)
    for tick, part in zip(ax.get_yticklabels(), parts_by_median):
        if part in PART_HIGHLIGHTS:
            tick.set_color(PART_CLASS_COLORS[PART_HIGHLIGHTS[part][0]])
        else:
            tick.set_color(TEXT_GREY)

    ax.set_xlabel(r"log$_2$ enrichment (top-5% / bot-5% by Circuit Score (log))",
                   fontsize=6.0, color=TEXT_GREY, labelpad=2.0)
    ax.set_xlim(left=l2_eligible["log_odds"].min() - 0.4,
                 right=l2_eligible["log_odds"].max() + 0.4)
    ax.set_ylim(-0.6, len(parts_by_median) - 0.4)
    ax.tick_params(axis="x", labelsize=5.5, labelcolor=TEXT_GREY,
                    color=DARK_GRAY, width=0.6, length=2.5)
    ax.tick_params(axis="y", color=DARK_GRAY, width=0.6, length=2.5)

    mid_y = len(parts_by_median) / 2
    ax.text(-3, mid_y, "destructive thr",
             color=PART_CLASS_COLORS["destructive_t2"],
             fontsize=5.5, ha="right", va="center",
             rotation=90, fontweight="normal",
             bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                        edgecolor="none", alpha=0.85), zorder=7)
    ax.text(1.5, mid_y, "constructive thr",
             color=PART_CLASS_COLORS["constructive"],
             fontsize=5.5, ha="left", va="center",
             rotation=90, fontweight="normal",
             bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                        edgecolor="none", alpha=0.85), zorder=7)

    fig.text(0.58, 0.975, "Per part circuit score probability enrichment",
              ha="center", va="top",
              fontsize=10, fontweight="normal", color=DARK_GRAY)

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
    leg = ax.legend(handles=handles, loc="upper left", frameon=False,
                     fontsize=5.5, handletextpad=0.4,
                     handlelength=1.0, handleheight=1.0,
                     labelspacing=0.5, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color(TEXT_GREY)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK_GRAY)
        ax.spines[spine].set_linewidth(0.6)

    out_base = PANELS_VEC / "panel_b_v2"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel B v2 written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
