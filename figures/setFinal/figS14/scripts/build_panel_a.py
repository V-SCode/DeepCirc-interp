"""Figure S05 — Panel A: per-part log-odds beeswarm across universal-eligible roles.

Faithful paper-pipeline port of `topology/figures/output/g3/
fig02_toxic_class` Panel A. Same data + same per-part strip-plot
layout; restyled to paper PASTEL palette and grey/non-bold labels.

Layout:
  y-axis = 20 library parts sorted by median log-odds (most-toxic at
           bottom, most-protective at top).
  x-axis = log-odds(top-5% / bot-5% by growth_score) for each
           (part, fp_key) cell that passes the universal-eligible
           filter (n_topologies >= 5, n_targets >= 3,
           total_designs >= 1000, task = toxicity).
  per-part dots = one per fp_key cell, jittered vertically.
  vertical reference lines at log_odds = 0, -2 (tier-2 threshold),
                                          -4 (extreme threshold).
  horizontal banding by toxic-class region (toxic_extreme +
  toxic_tier2 at the bottom, protective at the top).

Paper-style palette for the toxic / neutral / protective semantic
class encoding (replaces the topology family's saturated divergent
red$\\rightarrow$green ramp):
  toxic_extreme  $\\rightarrow$ PASTEL["red"]    (IcaRA/I1)
  toxic_tier2    $\\rightarrow$ PASTEL["orange"] (QacR/Q1, HlyIIR/H1)
  protective     $\\rightarrow$ PASTEL["green"]  (QacR/Q2, PhlF/P1)
  neutral        $\\rightarrow$ LIGHT_GRAY       (most parts)

Source: data/topology_g3/l2_top05/l2_enrichment.csv (narrower
top-5% / bot-5% tier cutoffs; sharper than the 25/25 version, paired
with the new figS06 circuit-side counterpart for visual parity).

Outputs:
  panels/vector/panel_a.pdf | .svg
  panels/raster/panel_a.png

Run:
    python \
        figures/setFinal/figS14/scripts/build_panel_a.py
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

# Paper PASTEL gate-class palette (ordinal: toxic $\\rightarrow$ neutral $\\rightarrow$ protective).
PART_CLASS_COLORS = {
    "toxic_extreme":  "#111111",   # near-black — strongest "toxic" emphasis
    "toxic_tier2":    MID_GRAY,    # "#9E9E9E" — second-tier toxic
    "protective":     ACCENT_BLUE,
    "neutral":        LIGHT_GRAY,
}

# Highlight categories — which parts get which class.
PART_HIGHLIGHTS = {
    "IcaRA/I1":   ("toxic_extreme", "extreme toxic"),
    "QacR/Q1":    ("toxic_tier2",   "tier-2 toxic"),
    "HlyIIR/H1":  ("toxic_tier2",   "tier-2 toxic"),
    "QacR/Q2":    ("protective",    "protective"),
    "PhlF/P1":    ("protective",    "protective"),
}


# --- canvas ----------------------------------------------------------------

CANVAS_W_MM = 200
CANVAS_H_MM = 165


def build_panel() -> None:
    use_style()

    l2 = pd.read_csv(L2_TOP05_CSV)
    l2_eligible = l2[
        (l2["task"] == "toxicity")
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
    toxic_idxs = [i for i, p in enumerate(parts_by_median)
                  if PART_HIGHLIGHTS.get(p, ("neutral",))[0] in
                  ("toxic_extreme", "toxic_tier2")]
    protective_idxs = [i for i, p in enumerate(parts_by_median)
                       if PART_HIGHLIGHTS.get(p, ("neutral",))[0] == "protective"]
    if toxic_idxs:
        # Toxic-region background tint swapped from pastel red $\\rightarrow$ DARK_GRAY
        # to align with paper palette (no warm tones in the bulk fills);
        # the per-part dots, tick labels, threshold markers + legend
        # keep their red/orange identity.
        ax.axhspan(min(toxic_idxs) - 0.5, max(toxic_idxs) + 0.5,
                    color=DARK_GRAY, alpha=0.10, zorder=0)
    if protective_idxs:
        ax.axhspan(min(protective_idxs) - 0.5, max(protective_idxs) + 0.5,
                    color=PART_CLASS_COLORS["protective"], alpha=0.12,
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

        # Median tick.
        med = np.median(x)
        ax.plot([med, med], [i - 0.40, i + 0.40],
                 color=DARK_GRAY, linewidth=0.9, zorder=6)

    # --- reference lines (rescaled for top-5/bot-5; wider distribution) -
    # Thresholds in log₂ space (rescaled 2026-06-01 to preserve the
    # semantic boundaries used pre-log₂-migration: tier-2 ≈ −3, extreme ≈ −6).
    ax.axvline(0, color=MID_GRAY, linestyle=":", linewidth=0.6, zorder=1)
    ax.axvline(-3, color=PART_CLASS_COLORS["toxic_tier2"], linestyle="--",
                linewidth=0.55, alpha=0.7, zorder=1)
    ax.axvline(-6, color=PART_CLASS_COLORS["toxic_extreme"], linestyle="--",
                linewidth=0.55, alpha=0.7, zorder=1)

    # --- y-axis labels --------------------------------------------------
    ax.set_yticks(range(len(parts_by_median)))
    ax.set_yticklabels(parts_by_median, fontsize=12)
    for tick, part in zip(ax.get_yticklabels(), parts_by_median):
        if part in PART_HIGHLIGHTS:
            tick.set_color(PART_CLASS_COLORS[PART_HIGHLIGHTS[part][0]])
        else:
            tick.set_color("#000000")

    ax.set_xlabel(r"log$_2$ enrichment (top-5% / bot-5% by growth score (raw))",
                   fontsize=12, color="#000000", labelpad=2.0)
    ax.set_xlim(left=l2_eligible["log_odds"].min() - 0.4,
                 right=l2_eligible["log_odds"].max() + 0.4)
    ax.set_ylim(-0.6, len(parts_by_median) - 0.4)
    ax.tick_params(axis="x", labelsize=12, labelcolor="#000000",
                    color="#000000", width=0.6, length=2.5)
    ax.tick_params(axis="y", color="#000000", width=0.6, length=2.5)

    # --- threshold labels ON the dashed lines mid-plot ------------------
    mid_y = len(parts_by_median) / 2
    ax.text(-3, mid_y, "tier-2 thr",
             color=PART_CLASS_COLORS["toxic_tier2"],
             fontsize=12, ha="right", va="center",
             rotation=90, fontweight="normal",
             bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                        edgecolor="none", alpha=0.85), zorder=7)
    ax.text(-6, mid_y, "extreme thr",
             color=PART_CLASS_COLORS["toxic_extreme"],
             fontsize=12, ha="right", va="center",
             rotation=90, fontweight="normal",
             bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                        edgecolor="none", alpha=0.85), zorder=7)

    # --- title ----------------------------------------------------------
    # Rendered via fig.text() (matches figS05 Panel D / E convention).
    # x=0.58 lines the title up with the axes centre (axes is at
    # left=0.21 width=0.74 $\\rightarrow$ centre 0.58); using fig-x=0.5 would
    # shift the title ~8% left of the plot content below.
    fig.text(0.58, 0.95, "Per part growth score probability enrichment",
              ha="center", va="top",
              fontsize=19, fontweight="normal", color="#000000")

    # --- legend ---------------------------------------------------------
    handles = [
        mpatches.Patch(color=PART_CLASS_COLORS["toxic_extreme"],
                        label="extreme toxic"),
        mpatches.Patch(color=PART_CLASS_COLORS["toxic_tier2"],
                        label="tier-2 toxic"),
        mpatches.Patch(color=PART_CLASS_COLORS["protective"],
                        label="protective"),
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

    out_base = PANELS_VEC / "panel_a"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel A written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
