"""DRAFT panel — position-dependence metrics table.

Sandbox table panel for figS05; NOT placed in the manifest.

Summarises the per-part position-dependence statistics computed from the
top-5% / bot-5% L2 enrichment (`data/G3/l2_top05/l2_enrichment.csv`)
across the 6 cascade-role buckets defined in Panel C
(NOT/NOR × early/middle/late).

Rows: 20 library parts, sorted by max(growth SD, circuit SD) descending
      — so the most position-dependent parts in EITHER metric land at
      the top.

Columns: per-task block of {mean log_odds, SD across buckets, range
across buckets}, with growth and circuit side-by-side.

Footer: aggregate row with mean SD, median SD, max SD, and the
variance-decomposition η² statistics that answer "how much of the
total log-odds variance is attributable to cascade position vs to
part identity".

Style: monospace numeric font, light banding, paper PASTEL green/red
on the per-part SD column to highlight position-dependent parts.

Outputs:
  panels/vector/panel_draft_position_dependence_table.pdf | .svg
  panels/raster/panel_draft_position_dependence_table.png

Run:
    python \\
        figures/setFinal/figS14/scripts/\\
        build_draft_position_dependence_table.py
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
)

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS14"
              / "panels" / "vector")

TEXT_GREY = "#666666"
BAND_BG   = "#F4F4F4"
HEADER_BG = "#E2E6EB"
FOOTER_BG = "#EAF1FA"


BUCKETS = ["NOT-early", "NOT-middle", "NOT-late",
           "NOR-early", "NOR-middle", "NOR-late"]


def bucket_of(row: pd.Series) -> str:
    nt = row["node_type"]
    try:
        d = int(row["depth_to_output"])
    except (TypeError, ValueError):
        d = -1
    if d == 1:
        return f"{nt}-late"
    if d == 2:
        return f"{nt}-middle"
    return f"{nt}-early"


def heatmap_grid(df: pd.DataFrame, task: str) -> pd.DataFrame:
    e = df[
        (df["task"] == task)
        & (df["n_topologies"] >= 5)
        & (df["n_targets"] >= 3)
        & (df["total_designs"] >= 1000)
    ].copy()
    e["bucket"] = e.apply(bucket_of, axis=1)
    g = (
        e.groupby(["part_name", "bucket"])["log_odds"]
        .mean()
        .unstack("bucket")
        .reindex(columns=BUCKETS)
    )
    return g


def per_part_stats(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for part in grid.index:
        v = grid.loc[part].values
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        rows.append({
            "part":   part,
            "mean":   float(v.mean()),
            "sd":     float(v.std(ddof=1)),
            "range":  float(v.max() - v.min()),
        })
    return pd.DataFrame(rows).set_index("part")


def eta_sq(grid: pd.DataFrame) -> dict:
    flat = grid.values[np.isfinite(grid.values)]
    grand = flat.mean()
    ss_total = float(((flat - grand) ** 2).sum())
    ss_between = 0.0
    for part in grid.index:
        v = grid.loc[part].values
        v = v[np.isfinite(v)]
        if v.size:
            ss_between += float(v.size * (v.mean() - grand) ** 2)
    ss_within = ss_total - ss_between
    return {
        "eta_sq_part":     ss_between / ss_total,
        "eta_sq_position": ss_within  / ss_total,
    }


# --- canvas ---------------------------------------------------------------

CANVAS_W_MM = 174
CANVAS_H_MM = 145


def build_panel() -> None:
    use_style()

    df = pd.read_csv(L2_TOP05_CSV)
    g_grid = heatmap_grid(df, "toxicity")
    c_grid = heatmap_grid(df, "circuit_log")
    g_stats = per_part_stats(g_grid)
    c_stats = per_part_stats(c_grid)
    g_eta = eta_sq(g_grid)
    c_eta = eta_sq(c_grid)

    # Combine into a single frame and sort by max(growth SD, circuit SD).
    both = pd.DataFrame({
        "g_mean":  g_stats["mean"],
        "g_sd":    g_stats["sd"],
        "g_range": g_stats["range"],
        "c_mean":  c_stats["mean"],
        "c_sd":    c_stats["sd"],
        "c_range": c_stats["range"],
    })
    both["sort_key"] = both[["g_sd", "c_sd"]].max(axis=1)
    both = both.sort_values("sort_key", ascending=False).drop(columns="sort_key")

    # --- layout grid in axes fraction (0..1) -----------------------------
    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Column x-edges. Layout: Part | g_mean | g_sd | g_range || c_mean | c_sd | c_range
    col_x = [0.02, 0.18, 0.30, 0.40, 0.50,
             0.62, 0.74, 0.84, 0.96]
    col_centers = [(col_x[i] + col_x[i + 1]) / 2 for i in range(len(col_x) - 1)]
    n_data_rows = len(both)

    # Row layout (y-fraction, growing downward from top).
    title_h    = 0.07
    super_h    = 0.05   # super-header strip (GROWTH | CIRCUIT)
    header_h   = 0.05
    data_h     = 0.55
    footer_h   = 0.28
    # Y boundaries (top-down):
    y_top         = 1.00
    y_super_top   = y_top - title_h
    y_super_bot   = y_super_top - super_h
    y_header_top  = y_super_bot
    y_header_bot  = y_header_top - header_h
    y_data_top    = y_header_bot
    y_data_bot    = y_data_top - data_h
    row_h         = data_h / n_data_rows
    y_footer_top  = y_data_bot
    y_footer_bot  = y_footer_top - footer_h

    # --- title ---------------------------------------------------------
    ax.text(0.02, (y_top + y_super_top) / 2,
             "DRAFT — per-part position-dependence: growth vs circuit",
             fontsize=7.2, color=TEXT_GREY, va="center", ha="left")
    ax.text(0.02, (y_top + y_super_top) / 2 - 0.022,
             "from L2 top-5% / bot-5% enrichment (G3 substrate, "
             "65 universal-eligible fp_keys, 6 cascade-role buckets)",
             fontsize=5.0, color=TEXT_GREY, va="center", ha="left")

    # --- super-header (GROWTH | CIRCUIT) -------------------------------
    # Growth block: cols 1..3 → x in [col_x[1], col_x[4]]
    # Circuit block: cols 4..6 → x in [col_x[5], col_x[8]]
    ax.add_patch(mpatches.Rectangle(
        (col_x[1], y_super_bot), col_x[4] - col_x[1], super_h,
        facecolor=PASTEL["green"], alpha=0.20, edgecolor="none"))
    ax.add_patch(mpatches.Rectangle(
        (col_x[5], y_super_bot), col_x[8] - col_x[5], super_h,
        facecolor=PASTEL["blue"], alpha=0.18, edgecolor="none"))
    ax.text((col_x[1] + col_x[4]) / 2, y_super_bot + super_h / 2,
             "GROWTH (toxicity)",
             fontsize=6.0, fontweight="bold", color=TEXT_GREY,
             ha="center", va="center")
    ax.text((col_x[5] + col_x[8]) / 2, y_super_bot + super_h / 2,
             "CIRCUIT (circuit_log)",
             fontsize=6.0, fontweight="bold", color=TEXT_GREY,
             ha="center", va="center")

    # --- column headers ------------------------------------------------
    ax.add_patch(mpatches.Rectangle(
        (0, y_header_bot), 1, header_h,
        facecolor=HEADER_BG, edgecolor="none"))
    headers = ["part", "mean", "SD", "range", "", "mean", "SD", "range"]
    for cx, h in zip(col_centers, headers):
        if not h:
            continue
        ax.text(cx, y_header_bot + header_h / 2, h,
                 fontsize=5.5, fontweight="bold", color=TEXT_GREY,
                 ha="center", va="center")

    # --- data rows -----------------------------------------------------
    g_sd_max  = float(both["g_sd"].max())
    c_sd_max  = float(both["c_sd"].max())

    def sd_color(sd_val: float, sd_max: float) -> str:
        """Light-green emphasis tint for above-median SD."""
        if sd_val >= 0.5 * sd_max:
            return PASTEL["green"]
        return "none"

    for i, (part, row) in enumerate(both.iterrows()):
        y_row_top = y_data_top - i * row_h
        y_row_bot = y_row_top - row_h
        # Band background.
        if i % 2 == 1:
            ax.add_patch(mpatches.Rectangle(
                (0, y_row_bot), 1, row_h,
                facecolor=BAND_BG, edgecolor="none", zorder=0))
        # SD-tint backgrounds.
        if row["g_sd"] >= 0.5 * g_sd_max:
            ax.add_patch(mpatches.Rectangle(
                (col_x[2], y_row_bot), col_x[3] - col_x[2], row_h,
                facecolor=PASTEL["green"], alpha=0.28,
                edgecolor="none", zorder=1))
        if row["c_sd"] >= 0.5 * c_sd_max:
            ax.add_patch(mpatches.Rectangle(
                (col_x[6], y_row_bot), col_x[7] - col_x[6], row_h,
                facecolor=PASTEL["green"], alpha=0.28,
                edgecolor="none", zorder=1))

        y_text = y_row_bot + row_h / 2
        # Part name (left-aligned).
        ax.text(col_x[0] + 0.005, y_text, part,
                 fontsize=5.0, color=TEXT_GREY, ha="left", va="center",
                 zorder=2)
        # Numeric columns. mean is signed (uses +/-), SD and range are
        # non-negative so we drop the sign for those.
        vals = [(row["g_mean"],  "signed"),
                (row["g_sd"],    "abs"),
                (row["g_range"], "abs"),
                (None,           None),
                (row["c_mean"],  "signed"),
                (row["c_sd"],    "abs"),
                (row["c_range"], "abs")]
        for cx, (v, kind) in zip(col_centers[1:], vals):
            if v is None:
                continue
            fmt = f"{v:+.2f}" if kind == "signed" else f"{v:.2f}"
            ax.text(cx, y_text, fmt,
                     fontsize=4.8, color=DARK_GRAY,
                     ha="center", va="center", family="monospace",
                     zorder=2)

    # --- footer (aggregate stats) -------------------------------------
    ax.add_patch(mpatches.Rectangle(
        (0, y_footer_bot), 1, footer_h,
        facecolor=FOOTER_BG, edgecolor="none"))

    footer_rows = [
        ("mean SD across parts",
         f"{g_stats['sd'].mean():.3f}",
         f"{c_stats['sd'].mean():.3f}"),
        ("median SD across parts",
         f"{g_stats['sd'].median():.3f}",
         f"{c_stats['sd'].median():.3f}"),
        ("max SD (most position-dependent part)",
         f"{g_stats['sd'].max():.3f}  ({g_stats['sd'].idxmax()})",
         f"{c_stats['sd'].max():.3f}  ({c_stats['sd'].idxmax()})"),
        ("η²_part   — variance from part identity",
         f"{g_eta['eta_sq_part']*100:.1f}%",
         f"{c_eta['eta_sq_part']*100:.1f}%"),
        ("η²_position — variance from cascade slot",
         f"{g_eta['eta_sq_position']*100:.1f}%",
         f"{c_eta['eta_sq_position']*100:.1f}%"),
    ]
    n_footer = len(footer_rows)
    foot_row_h = footer_h * 0.85 / n_footer
    foot_top   = y_footer_top - 0.012
    for i, (lbl, gv, cv) in enumerate(footer_rows):
        y = foot_top - (i + 0.5) * foot_row_h
        # Label (left).
        ax.text(col_x[0] + 0.005, y, lbl,
                 fontsize=5.0, color=TEXT_GREY,
                 ha="left", va="center")
        # Growth value (center of growth block).
        ax.text((col_x[1] + col_x[4]) / 2, y, gv,
                 fontsize=5.0, color=DARK_GRAY,
                 ha="center", va="center", family="monospace")
        # Circuit value (center of circuit block).
        ax.text((col_x[5] + col_x[8]) / 2, y, cv,
                 fontsize=5.0, color=DARK_GRAY,
                 ha="center", va="center", family="monospace")

    # Headline interpretation below the footer rows.
    ratio = c_eta["eta_sq_position"] / max(g_eta["eta_sq_position"], 1e-9)
    ax.text(0.5, y_footer_bot + 0.008,
             f"Position explains {c_eta['eta_sq_position']*100:.1f}% of "
             f"circuit variance vs {g_eta['eta_sq_position']*100:.1f}% of "
             f"growth variance ({ratio:.1f}× more position-dependent for "
             f"circuit).",
             fontsize=5.2, fontweight="bold", color="#1A3A5C",
             ha="center", va="bottom")

    # --- border lines (light, paper style) ----------------------------
    # Top + bottom of table region.
    for y in (y_top, y_super_bot, y_header_bot, y_data_bot, y_footer_bot):
        ax.plot([0, 1], [y, y], color=DARK_GRAY, linewidth=0.4, zorder=10)
    # Vertical divider between growth and circuit blocks (full height).
    ax.plot([col_x[4], col_x[4]],
             [y_super_bot, y_footer_bot],
             color=DARK_GRAY, linewidth=0.4, zorder=10)
    # Outer rect.
    ax.add_patch(mpatches.Rectangle(
        (0, y_footer_bot), 1, y_top - y_footer_bot,
        facecolor="none", edgecolor=DARK_GRAY, linewidth=0.5, zorder=10))

    out_base = PANELS_VEC / "panel_draft_position_dependence_table"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Draft position-dependence table written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
