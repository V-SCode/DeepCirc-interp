"""Figure S03 — Panel B: full design-score distribution (CIRCUIT, all sizes).

Single axis showing all three regulator-size curves (5/6/7-REG)
overlaid across the full rank-percentile range (0–100%). Direct
paper-pipeline port of the TOP graph of
``topology/figures/output/g3/fig31_v3_design_score_full_distribution``.

Each curve traces raw circuit_score vs rank percentile across the
size's complete design space (1003-anchor dense percentile sweep).
Log y-axis since the distribution spans ~6 orders of magnitude
(~0.04 RPU at p0.01 to ~2200 RPU at p99.99). Right-edge tier bands
mark the top 25 / 10 / 5 / 1% regions; paper Fig. 3 yellow-dot
anchors overlaid at their corresponding rank-percentile positions.

Pairs with Panel C (top-10% zoom). Total design counts shown in
upper-left legend; Panel C uses that space for per-tier counts.

Outputs (vector + raster):
  panels/vector/panel_b.pdf | .svg
  panels/raster/panel_b.png

Run:
    python \\
        figures/setFinal/figS10/scripts/build_panel_b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    PASTEL, DARK_GRAY, MID_GRAY, ACCENT_YELLOW, ACCENT_YELLOW_DARK, ACCENT_BLUE,
)

TEXT_GREY = "#666666"   # matches the figS04/S05 axis-label convention


# --- config ----------------------------------------------------------------

DATA_PATH = (REPO_ROOT / "data" / "topology_g3"
             / "size_tier_designs" / "dense_percentile_sweep.json")
PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS10"
              / "panels" / "vector")

SIZES = (5, 6, 7)

# Monochrome line-style mapping (replaces the previous green / orange
# / red SIZE_COLOR triplet so the panel reads in the paper's grey /
# blue identity). 5-reg solid, 6-reg dotted, 7-reg dashed.
SIZE_STYLE = {
    5: "-",
    6: ":",
    7: "--",
}
SIZE_LABEL = {s: f"{s} regulators" for s in SIZES}

# Tier bands at the chart edges. Top side (good) = pastel-green; bot
# side (bad) = pastel-red. Alpha ramps up toward the chart edges (the
# more extreme tier gets more saturated shading).
BAND_SPECS_TOP = [
    (90.0,  95.0, 0.16),
    (95.0,  99.0, 0.22),
    (99.0, 100.0, 0.35),
]
BAND_SPECS_BOT = [
    ( 0.0,  1.0, 0.35),   # bot 1% — most saturated, at the left edge
    ( 1.0,  5.0, 0.22),
    ( 5.0, 10.0, 0.16),
]

# Paper Fig. 3 yellow-dot anchors (sim-rank-1 of each exemplar).
# (size, target_name, circuit_score, growth_score)
PAPER_YELLOW_DOTS = {
    5: ("0x2B",  36.5869),
    6: ("0x17", 125.7974),
    7: ("0x6D",   5.2859),
}

# Native canvas matches the manifest slot (95 × 90 mm) so the panel
# renders 1:1 in the composed figure — title / axis labels / tick
# labels all appear at their declared point size (was 120 × 100, which
# got scaled to 0.79× when slotted, making everything visibly smaller
# than the equivalent text in figS03 Panel D + figS04 / figS05).
CANVAS_W_MM = 95
CANVAS_H_MM = 90


# --- helpers ---------------------------------------------------------------

def _format_count(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return f"{n:,}"


def _rank_pct_at_score(anchors: np.ndarray,
                        log_vec: np.ndarray,
                        target_score: float) -> float:
    """Inverse-interpolate: given a raw circuit_score, return the rank
    percentile where the size's CDF reaches that score (log-space)."""
    target_log = np.log2(max(target_score, 1e-9))
    return float(np.interp(target_log, log_vec, anchors))


def load_sweep() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


# --- panel assembly --------------------------------------------------------

def build_panel() -> None:
    use_style()

    sweep = load_sweep()
    anchors = np.array(sweep["anchors_pct"], dtype=float)

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes([0.14, 0.15, 0.83, 0.73])

    # Tier bands drawn first so curves overlay them. Top side (good) =
    # paper ACCENT_BLUE so the elite design-space region pops in the
    # DeepCirc signature cyan; bot side (bad) = DARK_GRAY (paper's
    # stroke/text colour) so the failure region is present but quiet.
    # Mirrors the asymmetric blue/grey emphasis used across figS05.
    for x0, x1, alpha in BAND_SPECS_TOP:
        ax.axvspan(x0, x1, color=ACCENT_BLUE, alpha=alpha, zorder=0)
    for x0, x1, alpha in BAND_SPECS_BOT:
        ax.axvspan(x0, x1, color=DARK_GRAY, alpha=alpha, zorder=0)

    # Track per-size totals for the legend.
    legend_rows = []
    all_y = []
    yellow_dot_positions: list[tuple[int, float, float]] = []
    for s in SIZES:
        size_data = sweep["per_size"][str(s)]
        log_vec = np.array(size_data["circuit_log"], dtype=float)
        raw = 2.0 ** log_vec   # log_vec is log₂ since the 2026-06-01 migration
        all_y.extend(raw.tolist())

        linestyle = SIZE_STYLE[s]
        ax.plot(anchors, raw, linestyle=linestyle, color=DARK_GRAY,
                  linewidth=1.5, alpha=0.95, zorder=3,
                  solid_capstyle="round", dash_capstyle="round")

        n_total = size_data["n_designs_total_circuit"]
        legend_rows.append((linestyle, SIZE_LABEL[s],
                              _format_count(n_total)))

        # Yellow-dot anchor: locate the rank-percentile where this
        # size's curve reaches the paper-anchor circuit_score.
        if s in PAPER_YELLOW_DOTS:
            _name, anchor_score = PAPER_YELLOW_DOTS[s]
            pct = _rank_pct_at_score(anchors, log_vec, anchor_score)
            yellow_dot_positions.append((s, pct, anchor_score))

    # Log y-axis (~6 orders of magnitude across the full sweep).
    ax.set_yscale("log")
    y_min = max(min(y for y in all_y if y > 0), 1e-3)
    y_max = max(all_y)
    log_min = np.log10(y_min)
    log_max = np.log10(y_max)
    log_range = log_max - log_min
    ax.set_ylim(10 ** (log_min - 0.04 * log_range),
                  10 ** (log_max + 0.20 * log_range))

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("Rank Percentile",
                   fontsize=8, color="#000000", labelpad=2.0)
    ax.set_ylabel("Circuit Score (raw)",
                   fontsize=8, color="#000000", labelpad=2.0)
    ax.tick_params(axis="x", labelsize=8, labelcolor="#000000",
                    color="#000000", width=0.6, length=2.5)
    ax.tick_params(axis="y", labelsize=8, labelcolor="#000000",
                    color="#000000", width=0.6, length=2.5)

    ax.grid(True, which="major", linewidth=0.4, alpha=0.35,
              color="#bbb", zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (Yellow-dot anchor markers removed per user direction — keep
    # the chart focused on the size-distribution curves themselves.)

    # Legend at upper-CENTRE — vertical stack (one row per size) with
    # a short line-style sample next to each label so the
    # solid/dotted/dashed mapping reads at a glance. ncol=1 keeps the
    # entries from overlapping the curves or the tier-band labels at
    # either chart edge.
    proxies = [
        Line2D([], [], color=DARK_GRAY, linestyle=ls, linewidth=1.5,
                 solid_capstyle="round", dash_capstyle="round")
        for ls, _lbl, _cnt in legend_rows
    ]
    labels = [f"{lbl} ,  {cnt} designs"
                for _ls, lbl, cnt in legend_rows]
    leg = ax.legend(proxies, labels,
                       loc="upper center", ncol=1,
                       bbox_to_anchor=(0.5, 0.99),
                       fontsize=7, frameon=False,
                       handlelength=1.8, handletextpad=0.5,
                       labelspacing=0.35, borderpad=0.1)
    for txt in leg.get_texts():
        txt.set_color("#000000")

    # Tier-band labels — rotated, anchored at the chart's top edge so
    # both top- and bot-side labels occupy the same vertical band.
    #
    # Reading direction is mirrored across the two sides:
    #   * top labels rotate CCW (+90) $\\rightarrow$ letters read bottom-to-top
    #   * bot labels rotate CW (-90)  $\\rightarrow$ letters read top-to-bottom
    #
    # Both bboxes anchor at the TOP of the rotated text (va="top") so
    # text extends DOWN into the chart from label_y. ha mirrors across
    # x=50 so each label sits OUTSIDE its band, away from the chart
    # edge:
    #   * top labels — rotation=+90, va="top", ha="right"
    #                  (anchor LEFT of band's left boundary;
    #                   text extends LEFT of anchor, further from band)
    #   * bot labels — rotation=-90, va="top", ha="left"
    #                  (anchor RIGHT of band's right boundary;
    #                   text extends RIGHT of anchor, further from band)
    band_labels = [
        # (x_anchor, label, ha, rotation, va)
        ( 1.6, "bot 1%",  "left",  -90, "top"),
        ( 5.6, "bot 5%",  "left",  -90, "top"),
        (10.6, "bot 10%", "left",  -90, "top"),
        (89.4, "top 10%", "right",  90, "top"),
        (94.4, "top 5%",  "right",  90, "top"),
        (98.4, "top 1%",  "right",  90, "top"),
    ]
    label_y = 10 ** (log_max + 0.17 * log_range)
    for x_anchor, lbl, ha, rot, va in band_labels:
        ax.text(x_anchor, label_y, lbl,
                  ha=ha, va=va, fontsize=8,
                  color="#666", style="italic", rotation=rot,
                  zorder=4)

    # Title — DeepCirc paper formatting, 7pt bold black, centred above
    # the axes block (matches Panel C convention).
    fig.text(0.55, 0.965,
              "Full design-score distribution by circuit size",
              ha="center", va="top",
              fontsize=9, fontweight="normal", color="#000000")

    out_base = PANELS_VEC / "panel_b"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel B written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
