"""Companion table for figS04 Panel C — part-frequency counts.

Counts how often each TetR-library part appears across the 30 designs
in the new Panel C table (15 max-circuit + 15 pareto-optimal, both
deduplicated to 1 per target function at paper MLP-gate thresholds).
Each row shows
the per-group counts AND the combined total so the contrast between
the two selection rules is visible side-by-side.

Top-5 frequent parts (PhlF/P2, SrpR/S4, AmtR/A1, BetI/E1, QacR/Q2) get
the same coloured-bold rendering as the canonical Panel C so a reader
can scan across panels.

Source data:
  * MAX-CIRCUIT top-15: per-target best by circuit_log, from
    all_topology_fronts.csv.gz filtered by paper MLP gates.
  * PARETO-OPTIMAL top-15: per-target best by combined_score (paper-
    MLP-gate floors), from knee_designs.csv (215 pareto-optimal
    designs filtered + scored). Internal variable names retain the
    legacy "knee" tag for back-compat with sibling scripts.

Outputs:
  final/panel_c_part_counts.pdf | .png
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(SCRIPT_PATH.parent))
import build_panel_c as bc  # noqa: E402  — reuse top-15 selection helpers

from figures.figtools import use_style, figsize_mm  # noqa: E402

FINAL_DIR = (PKG_ROOT / "figures" / "setFinal" / "figS13" / "final")

TOP5_COLORS = bc.TOP5_COLORS

# Full 20-part TetR library (paper Methods).
LIBRARY_PARTS = [
    "AmeR/F1", "AmtR/A1",
    "BM3R1/B1", "BM3R1/B2", "BM3R1/B3",
    "BetI/E1",
    "HlyIIR/H1",
    "IcaRA/I1",
    "LitR/L1",
    "LmrA/N1",
    "PhlF/P1", "PhlF/P2", "PhlF/P3", "PhlF/P4",
    "PsrA/R1",
    "QacR/Q1", "QacR/Q2",
    "SrpR/S1", "SrpR/S2", "SrpR/S3", "SrpR/S4",
]


def _count_in_set(df) -> tuple[Counter, int]:
    counts: Counter = Counter()
    slots = 0
    for parts_str in df["part_names"]:
        parts = (ast.literal_eval(parts_str)
                 if isinstance(parts_str, str) else list(parts_str))
        counts.update(parts)
        slots += len(parts)
    return counts, slots


def build_table() -> None:
    use_style()

    maxc15 = bc._top15_maxcirc_per_target()
    knee15 = bc._top15_knee_per_target()

    counts_max,  slots_max  = _count_in_set(maxc15)
    counts_knee, slots_knee = _count_in_set(knee15)

    n_max  = len(maxc15)
    n_knee = len(knee15)
    n_total = n_max + n_knee
    total_slots = slots_max + slots_knee

    counts_total = Counter()
    for p in LIBRARY_PARTS:
        counts_total[p] = counts_max.get(p, 0) + counts_knee.get(p, 0)

    rows_all = [(p, counts_max.get(p, 0), counts_knee.get(p, 0),
                 counts_total[p])
                for p in LIBRARY_PARTS]
    present = sorted([r for r in rows_all if r[3] > 0],
                     key=lambda r: (-r[3], r[0]))
    absent  = sorted([r[0] for r in rows_all if r[3] == 0])

    # ----- layout -----
    CANVAS_W_MM = 165.0   # wider than before — 6 cols instead of 5
    CANVAS_H_MM = 150.0

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))

    fig.text(0.5, 0.96,
              "Part frequency across Panel C top-15 max-circuit + top-15 pareto-optimal designs",
              ha="center", va="top",
              fontsize=8.5, fontweight="bold", color="#111")
    fig.text(0.5, 0.925,
              f"Per-target best designs (n={n_total} = {n_max} max-circuit + "
              f"{n_knee} pareto-optimal); paper MLP gates: circuit_score >= 2, growth >= 0.5",
              ha="center", va="top",
              fontsize=6.2, color="#444")

    LEFT   = 0.05
    RIGHT  = 0.975
    TOP    = 0.88
    BOTTOM = 0.22

    # Shifted "max" + "knee" left slightly so the renamed
    # "Pareto-Optimal (/ 15)" column header (longer than "Knee (/ 15)")
    # fits without overflowing into the Total column.
    col_x = {
        "rank":   LEFT + 0.005,
        "part":   LEFT + 0.060,
        "max":    LEFT + 0.275,
        "knee":   LEFT + 0.430,
        "total":  LEFT + 0.665,
        "pct":    LEFT + 0.825,
    }

    def hline(y, lw=0.6, color="#111"):
        line = Line2D([LEFT, RIGHT], [y, y],
                       transform=fig.transFigure,
                       color=color, linewidth=lw,
                       solid_capstyle="butt")
        fig.add_artist(line)

    hline(TOP + 0.005, lw=1.0)

    # Header
    header_y = TOP - 0.012
    headers = [
        ("rank",  "Rank"),
        ("part",  "Part"),
        ("max",   f"Max-Circ (/ {n_max})"),
        ("knee",  f"Pareto-Optimal (/ {n_knee})"),
        ("total", f"Total (/ {n_total})"),
        ("pct",   "% of 30"),
    ]
    for key, label in headers:
        fig.text(col_x[key], header_y, label,
                  fontsize=6.8, fontweight="bold", color="#111",
                  ha="left", va="top", transform=fig.transFigure)
    hline(TOP - 0.030, lw=0.6, color="#444")

    # Body rows
    n_rows = len(present)
    row_top = TOP - 0.046
    row_bot = BOTTOM
    row_h = (row_top - row_bot) / max(n_rows, 1)

    for i, (part, c_max, c_knee, c_tot) in enumerate(present):
        y = row_top - i * row_h - row_h / 2.0
        pct = 100.0 * c_tot / n_total

        is_top5 = part in TOP5_COLORS
        if is_top5:
            text_color = TOP5_COLORS[part]
            weight = "bold"
        else:
            text_color = "#111" if c_tot >= n_total * 0.5 else "#555"
            weight = "normal"

        fig.text(col_x["rank"], y, f"{i + 1}",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight="normal", transform=fig.transFigure)
        fig.text(col_x["part"], y, part,
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight=weight, family="monospace",
                  transform=fig.transFigure)
        fig.text(col_x["max"], y, f"{c_max}",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight=weight, family="monospace",
                  transform=fig.transFigure)
        fig.text(col_x["knee"], y, f"{c_knee}",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight=weight, family="monospace",
                  transform=fig.transFigure)
        fig.text(col_x["total"], y, f"{c_tot}",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight=weight, family="monospace",
                  transform=fig.transFigure)
        fig.text(col_x["pct"], y, f"{pct:.0f}%",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  family="monospace", transform=fig.transFigure)

    hline(BOTTOM - 0.005, lw=0.6, color="#444")

    # Footer: absent parts
    footer_y = BOTTOM - 0.030
    if absent:
        fig.text(LEFT, footer_y,
                  f"Library parts never appearing in either top-15 "
                  f"({len(absent)} / {len(LIBRARY_PARTS)}):",
                  fontsize=6.2, color="#444",
                  ha="left", va="top", transform=fig.transFigure)
        fig.text(LEFT, footer_y - 0.030,
                  "   " + "   ".join(absent),
                  fontsize=6.2, color="#666",
                  ha="left", va="top", family="monospace",
                  transform=fig.transFigure)

    fig.text(LEFT, footer_y - 0.085,
              f"Total slots filled: {slots_max} (max-circ) + {slots_knee} (pareto-optimal) "
              f"= {total_slots} across {n_total} designs.",
              fontsize=5.8, color="#888",
              ha="left", va="top", transform=fig.transFigure)

    hline(0.025, lw=1.0)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FINAL_DIR / "panel_c_part_counts.png"
    out_pdf = FINAL_DIR / "panel_c_part_counts.pdf"
    fig.savefig(out_png, dpi=600)
    from PIL import Image
    Image.open(out_png).convert("RGB").save(out_pdf, "PDF",
                                              resolution=600.0)
    plt.close(fig)
    print("Part-count table written:")
    print(f"  pdf: {out_pdf.relative_to(REPO_ROOT)}")
    print(f"  png: {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_table()
