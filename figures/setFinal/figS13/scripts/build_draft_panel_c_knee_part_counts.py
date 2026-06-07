"""DRAFT — part-frequency table for the knee-design Panel C variant.

Sandbox companion to `build_draft_panel_c_knee.py`. Mirrors
`build_panel_c_part_counts.py` exactly, but the source set is the
**knee** design per target (knee criterion A) instead of the
highest-circuit_score design per target. Both filters use the same
paper MLP-gate thresholds (circuit_score >= 2, growth >= 0.5).

Counts how often each TetR-library part appears across the per-target
knee designs. Renders the same three-column ranking layout used by the
canonical Panel C part-counts companion. Parts that never appear in
the knee set are listed in a footer.

Outputs:
  panels/vector/panel_draft_c_knee_part_counts.pdf | .png

NOT placed in the figS04 manifest — sandbox draft only.

Run:
    python \
        figures/setFinal/figS13/scripts/\
        build_draft_panel_c_knee_part_counts.py
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "topology" / "figures"))
import _loaders as L  # noqa: E402

from figures.figtools import use_style, figsize_mm  # noqa: E402

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS13"
              / "panels" / "vector")
PANELS_RAS = (PKG_ROOT / "figures" / "setFinal" / "figS13"
              / "panels" / "raster")

CIRC_SCORE_FLOOR = 2.0
TOX_FLOOR        = 0.5

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


def _parse_source_list(src) -> list[str]:
    s = str(src).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            return [s]
    return [s]


def _knee_per_target() -> pd.DataFrame:
    """Return one knee row per target — the topology whose knee_A has
    the highest circuit_log among all topologies serving that target,
    subject to the paper MLP-gate floors."""
    knees = L.load_pareto_knees("G3").copy()
    knees["circuit_log"]   = knees["knee_A_circuit_log"]
    knees["circuit_score"] = np.exp(knees["knee_A_circuit_log"])
    knees["toxicity"]      = knees["knee_A_toxicity"]
    knees["part_names"]    = knees["knee_A_part_names"]

    mask = ((knees["circuit_score"] >= CIRC_SCORE_FLOOR)
            & (knees["toxicity"]    >= TOX_FLOOR))
    buildable = knees[mask].copy()
    buildable["sources_list"] = buildable["source"].apply(_parse_source_list)

    rows: list[dict] = []
    for _, r in buildable.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)
    best = (long.sort_values("circuit_log", ascending=False)
                .groupby("target", as_index=False)
                .first())
    return best


def build_table() -> None:
    use_style()

    best = _knee_per_target()
    n_designs = len(best)

    counts: Counter = Counter()
    total_slots = 0
    for parts_str in best["part_names"]:
        parts = ast.literal_eval(parts_str)
        counts.update(parts)
        total_slots += len(parts)

    rows = [(p, counts.get(p, 0)) for p in LIBRARY_PARTS]
    present = sorted([r for r in rows if r[1] > 0], key=lambda r: (-r[1], r[0]))
    absent  = sorted([r[0] for r in rows if r[1] == 0])

    CANVAS_W_MM = 130.0
    CANVAS_H_MM = 145.0

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))

    fig.text(0.5, 0.96,
              "DRAFT — Part frequency across per-target knee designs",
              ha="center", va="top",
              fontsize=8.5, fontweight="bold", color="#111")
    fig.text(0.5, 0.925,
              f"Knee design per target function (criterion A, n={n_designs}); "
              f"paper MLP gates: circuit_score >= 2, growth >= 0.5",
              ha="center", va="top",
              fontsize=6.2, color="#444")

    LEFT   = 0.06
    RIGHT  = 0.97
    TOP    = 0.88
    BOTTOM = 0.22

    col_x = {
        "rank":   LEFT + 0.005,
        "part":   LEFT + 0.110,
        "count":  LEFT + 0.430,
        "pres":   LEFT + 0.580,
        "slot":   LEFT + 0.780,
    }

    def hline(y, lw=0.6, color="#111"):
        line = Line2D([LEFT, RIGHT], [y, y],
                       transform=fig.transFigure,
                       color=color, linewidth=lw,
                       solid_capstyle="butt")
        fig.add_artist(line)

    hline(TOP + 0.005, lw=1.0)

    header_y = TOP - 0.012
    headers = [
        ("rank",  "Rank"),
        ("part",  "Part"),
        ("count", f"Count (/ {n_designs})"),
        ("pres",  "% of designs"),
        ("slot",  "% of slots"),
    ]
    for key, label in headers:
        fig.text(col_x[key], header_y, label,
                  fontsize=6.8, fontweight="bold", color="#111",
                  ha="left", va="top", transform=fig.transFigure)
    hline(TOP - 0.030, lw=0.6, color="#444")

    n_rows = len(present)
    row_top = TOP - 0.046
    row_bot = BOTTOM
    row_h = (row_top - row_bot) / max(n_rows, 1)

    for i, (part, cnt) in enumerate(present):
        y = row_top - i * row_h - row_h / 2.0
        pres_pct = 100.0 * cnt / n_designs
        slot_pct = 100.0 * cnt / total_slots if total_slots else 0.0
        bold = "bold" if cnt == n_designs else "normal"
        text_color = "#111" if cnt >= n_designs * 0.5 else "#444"

        fig.text(col_x["rank"], y, f"{i + 1}",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight="normal", transform=fig.transFigure)
        fig.text(col_x["part"], y, part,
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight=bold, family="monospace",
                  transform=fig.transFigure)
        fig.text(col_x["count"], y, f"{cnt}",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  fontweight=bold, family="monospace",
                  transform=fig.transFigure)
        fig.text(col_x["pres"], y, f"{pres_pct:.0f}%",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  family="monospace", transform=fig.transFigure)
        fig.text(col_x["slot"], y, f"{slot_pct:.1f}%",
                  fontsize=6.5, color=text_color, ha="left", va="center",
                  family="monospace", transform=fig.transFigure)

    hline(BOTTOM - 0.005, lw=0.6, color="#444")

    footer_y = BOTTOM - 0.030
    if absent:
        fig.text(LEFT, footer_y,
                  f"Library parts never appearing in the knee-per-target set "
                  f"({len(absent)} / {len(LIBRARY_PARTS)}):",
                  fontsize=6.2, color="#444",
                  ha="left", va="top", transform=fig.transFigure)
        fig.text(LEFT, footer_y - 0.035,
                  "   " + "   ".join(absent),
                  fontsize=6.2, color="#666",
                  ha="left", va="top", family="monospace",
                  transform=fig.transFigure)

    fig.text(LEFT, footer_y - 0.105,
              f"Total slots filled across {n_designs} knee designs: {total_slots}",
              fontsize=5.8, color="#888",
              ha="left", va="top", transform=fig.transFigure)

    hline(0.025, lw=1.0)

    PANELS_VEC.mkdir(parents=True, exist_ok=True)
    PANELS_RAS.mkdir(parents=True, exist_ok=True)
    out_png = PANELS_RAS / "panel_draft_c_knee_part_counts.png"
    out_pdf = PANELS_VEC / "panel_draft_c_knee_part_counts.pdf"
    fig.savefig(out_png, dpi=600)
    from PIL import Image
    Image.open(out_png).convert("RGB").save(out_pdf, "PDF",
                                              resolution=600.0)
    plt.close(fig)
    print("Draft knee part-count table written:")
    print(f"  pdf: {out_pdf.relative_to(REPO_ROOT)}")
    print(f"  png: {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_table()
