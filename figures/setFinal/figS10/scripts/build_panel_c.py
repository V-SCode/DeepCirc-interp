"""Figure S03 — Panel C: top-10% design-score overlay (CIRCUIT, all sizes).

Single axis showing all three regulator-size curves (5/6/7-REG)
overlaid on the same top-10%-rank-percentile region. Direct paper-
pipeline port of the BOTTOM 3 zoom-in plots from
``topology/figures/output/g3/fig31_v3_design_score_full_distribution``
— but combined into ONE axis instead of three separate sub-panels.

Each curve traces raw circuit_score vs rank percentile across the
top-10% of the size's design space (anchors from p90 to p99.99). Log
y-axis since the elite tier spans ~3 orders of magnitude (p90 ~ 2 RPU
to p99.99 ~ 2200 RPU at 5-REG).

Outputs (vector + raster):
  panels/vector/panel_c.pdf | .svg
  panels/raster/panel_c.png

Run:
    python \\
        figures/setFinal/figS10/scripts/build_panel_c.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]   # .../DeepCirc-interp (repo root)
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import PASTEL, DARK_GRAY  # noqa: E402

TEXT_GREY = "#666666"   # matches the figS04/S05 axis-label convention


# --- config ----------------------------------------------------------------

DATA_PATH = (REPO_ROOT / "data" / "topology_g3"
             / "size_tier_designs" / "dense_percentile_sweep.json")
PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS10"
              / "panels" / "vector")

SIZES = (5, 6, 7)

# Monochrome line-style mapping (matches Panel B). All curves drawn
# in DARK_GRAY; size identity carried by linestyle.
SIZE_STYLE = {
    5: "-",
    6: ":",
    7: "--",
}
SIZE_LABEL = {s: f"{s} regulators:" for s in SIZES}

# Top-10% tier markers — shared across all 3 curves on the overlay
ZOOM_TIERS = [
    (90.0, "top 10%"),
    (95.0, "top 5%"),
    (99.0, "top 1%"),
]

# Canvas at a natural aspect for the overlay (composed slot is 92×90 mm;
# pick something close so the standalone preview is representative).
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


def load_sweep() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


# --- panel assembly --------------------------------------------------------

def build_panel() -> None:
    use_style()

    sweep = load_sweep()
    anchors = np.array(sweep["anchors_pct"], dtype=float)
    mask = anchors >= 90.0
    x = anchors[mask]

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes([0.14, 0.15, 0.83, 0.73])

    # Track per-tier design counts for the legend table.
    # (top-10% / top-5% / top-1% = 10% / 5% / 1% of total designs)
    tier_counts: dict[int, dict[str, str]] = {}
    all_y = []
    for s in SIZES:
        size_data = sweep["per_size"][str(s)]
        log_vec = np.array(size_data["circuit_log"], dtype=float)
        raw = 2.0 ** log_vec   # log_vec is log₂ since the 2026-06-01 migration
        y = raw[mask]
        all_y.extend(y.tolist())

        linestyle = SIZE_STYLE[s]
        ax.plot(x, y, linestyle=linestyle, color=DARK_GRAY,
                  linewidth=1.7, alpha=0.95, zorder=3,
                  solid_capstyle="round", dash_capstyle="round")

        n_designs = size_data["n_designs_total_circuit"]
        tier_counts[s] = {
            "top 10%": _format_count(int(round(n_designs * 0.10))),
            "top 5%":  _format_count(int(round(n_designs * 0.05))),
            "top 1%":  _format_count(int(round(n_designs * 0.01))),
        }

    # Log y-axis since top-10% region spans ~3 orders of magnitude.
    # Top padding kept at 0.22 × log_range so no new decade-gridline
    # (10^4) appears above the data — the design-count table is
    # positioned to fit inside the existing axis range, above the
    # 10^3 gridline.
    ax.set_yscale("log")
    y_min, y_max = min(all_y), max(all_y)
    log_min = np.log10(y_min)
    log_max = np.log10(y_max)
    log_range = log_max - log_min
    ax.set_ylim(10 ** (log_min - 0.04 * log_range),
                  10 ** (log_max + 0.22 * log_range))

    # Tier markers — vertical dashed lines at 90 / 95 / 99 (text
    # labels removed; the percentile axis ticks already mark the
    # boundaries, and removing the labels keeps the legend uncluttered).
    ax.set_xlim(89.5, 100.5)
    for p, _lbl in ZOOM_TIERS:
        ax.axvline(p, color="#888", linestyle="--", linewidth=0.6,
                    alpha=0.7, zorder=1)

    # Axes ticks + labels — aligned to figS04/S05 convention
    # (xlabel/ylabel 6.0 pt TEXT_GREY, tick labels 5.5 pt).
    ax.set_xticks([90, 95, 100])
    ax.set_xticklabels(["90%", "95%", "100%"], fontsize=8)
    ax.set_xlabel("Rank Percentile",
                   fontsize=8, color="#000000", labelpad=2.0)
    ax.set_ylabel("Circuit Score (raw)",
                   fontsize=8, color="#000000", labelpad=2.0)
    ax.tick_params(axis="x", labelsize=8, labelcolor="#000000",
                    color="#000000", width=0.6, length=2.5)
    ax.tick_params(axis="y", labelsize=8, labelcolor="#000000",
                    color="#000000", width=0.6, length=2.5)

    # Spines + grid
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35,
              color="#bbb", zorder=0)
    for spine in ax.spines.values():
        spine.set_color(DARK_GRAY)
        spine.set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Per-size × per-tier design-count table in upper-left (curves
    # rise to the right, so this corner stays clear). Header row of
    # tier labels in grey + 3 data rows in size colors. Avoids
    # repeating the total-design counts that will appear in Panel B.
    #
    # Positions in axes-fraction coords. Data xlim is (89.5, 100.5);
    # axes_x = (data_x - 89.5) / 11. Size labels left-aligned just
    # right of the 90% vertical line (small horizontal padding);
    # tier values centred between the dashed vertical lines at
    # 90 / 95 / 99 / right edge (100). Top of table sits at
    # axes-frac 0.92 with row spacing of 0.075 so rows don't bump
    # into log gridlines below.
    table_x_size  = 0.065     # data x=90 + small horizontal padding
    table_x_t10   = 0.273     # centred between 90 and 95
    table_x_t5    = 0.682     # centred between 95 and 99
    table_x_t1    = 0.909     # centred between 99 and 100
    table_y_top   = 0.94      # axes-frac y of header row
    # 4 rows × 0.055 spacing = bottom row at 0.94 - 3×0.055 = 0.775.
    # The 10^3 gridline sits at axes-frac ~0.73 under the existing
    # 0.22 top-padding ylim — so all rows fit comfortably above it
    # with no decade-gridline collisions.
    row_dy        = 0.055

    # Header row (grey, italic, smaller) — 7pt so the right-edge
    # "top 1%" label clears the chart's vertical tier-line cluster.
    ax.text(table_x_t10, table_y_top, "top 10%",
              transform=ax.transAxes, ha="center", va="top",
              fontsize=7, color="#666", style="italic", zorder=5)
    ax.text(table_x_t5, table_y_top, "top 5%",
              transform=ax.transAxes, ha="center", va="top",
              fontsize=7, color="#666", style="italic", zorder=5)
    ax.text(table_x_t1, table_y_top, "top 1%",
              transform=ax.transAxes, ha="center", va="top",
              fontsize=7, color="#666", style="italic", zorder=5)

    # Data rows: one per size. Short linestyle sample sits just LEFT
    # of the row label so the solid/dotted/dashed mapping is visible
    # without colour-coding the row. All text in "#000000".
    glyph_x0 = table_x_size - 0.055   # leftmost edge of the line sample
    glyph_x1 = table_x_size - 0.015   # rightmost edge (~5% of axis wide)
    # Vertical centring offset: the text uses va="top" with a fontsize
    # ~7pt, so the sample line sits ~half-line-height below table_y_top
    # to align with the text baseline midpoint.
    glyph_y_offset = -0.022

    for i, s in enumerate(SIZES):
        row_y = table_y_top - (i + 1) * row_dy
        # Line-style sample.
        ax.plot([glyph_x0, glyph_x1],
                  [row_y + glyph_y_offset, row_y + glyph_y_offset],
                  transform=ax.transAxes,
                  linestyle=SIZE_STYLE[s], color=DARK_GRAY,
                  linewidth=1.4, solid_capstyle="round",
                  dash_capstyle="round", zorder=5,
                  clip_on=False)
        # Normal-weight DARK_GRAY everywhere (matches Panel B's
        # legend convention; the row's line-style glyph carries the
        # 5/6/7-reg identity, so no need to bold the label or counts).
        ax.text(table_x_size, row_y, SIZE_LABEL[s],
                  transform=ax.transAxes, ha="left", va="top",
                  fontsize=7, color="#000000", zorder=5)
        ax.text(table_x_t10, row_y, tier_counts[s]["top 10%"],
                  transform=ax.transAxes, ha="center", va="top",
                  fontsize=7, color="#000000", zorder=5)
        ax.text(table_x_t5, row_y, tier_counts[s]["top 5%"],
                  transform=ax.transAxes, ha="center", va="top",
                  fontsize=7, color="#000000", zorder=5)
        ax.text(table_x_t1, row_y, tier_counts[s]["top 1%"],
                  transform=ax.transAxes, ha="center", va="top",
                  fontsize=7, color="#000000", zorder=5)

    # Title — DeepCirc paper formatting, 7pt bold black, centred above
    # the axes block.
    fig.text(0.55, 0.965,
              "Design-score distribution in the elite tier",
              ha="center", va="top",
              fontsize=9, fontweight="normal", color="#000000")

    out_base = PANELS_VEC / "panel_c"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel C written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
