"""Figure S04 — Panel B (split): one PDF per NPN class.

Renders the same 9 NPN-class Pareto cells that `build_panel_b.py` tiles
into the staggered 5-4 brick, but emits them as **9 standalone PDFs**
(plus SVG + PNG companions) so they can be individually placed and
rearranged in Illustrator.

Each cell is rendered at its native cell size (48 × 50 mm) — identical
to its tile size in the composed Panel B. Content + styling are
identical to `build_panel_b.py`: buildable region shading, per-topology
Pareto lines, knee markers, paper Fig. 3 yellow-dot ⭐ overlays,
per-cell target legend, grey/non-bold cell title, paper PASTEL palette.

Outputs:
  panels/vector/per_npn/panel_b_npn_<class>.{pdf,svg}
  panels/raster/per_npn/panel_b_npn_<class>.png

Run:
    python \
        figures/setFinal/figS12/scripts/build_panel_b_per_npn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    ACCENT_YELLOW, ACCENT_YELLOW_DARK, DARK_GRAY, PASTEL,
)


PANELS_VEC_DIR = (PKG_ROOT / "figures" / "setFinal" / "figS12"
                  / "panels" / "vector" / "per_npn")
PANELS_VEC_DIR.mkdir(parents=True, exist_ok=True)
(PANELS_VEC_DIR.parent.parent / "raster" / "per_npn").mkdir(parents=True, exist_ok=True)

TEXT_GREY = "#666666"

PAPER_YELLOW_DOTS_BY_NPN = {
    "0x17": [
        {"target": "0x2B", "size": 5, "circuit": 36.5869,  "growth": 0.7609},
        {"target": "0x17", "size": 6, "circuit": 125.7974, "growth": 0.7590},
    ],
    "0x16": [
        {"target": "0x6D", "size": 7, "circuit": 5.2859,   "growth": 0.6883},
    ],
}
for _yds in PAPER_YELLOW_DOTS_BY_NPN.values():
    for _yd in _yds:
        _yd["circuit_log"] = float(np.log(_yd["circuit"]))


# Native cell dimensions — match the staggered brick in build_panel_b.py
# so a placed panel_b_npn_<class>.pdf in Illustrator visually matches
# what the master Panel B shows.
CELL_W_MM = 48.0
CELL_H_MM = 50.0


def _render_one(ax, *, npn: str, sub_fronts: pd.DataFrame, sub_knees: pd.DataFrame,
                circ_floor: float, tox_floor: float, xlo: float, xhi: float,
                ylo: float, yhi: float) -> None:
    targets = sorted(sub_knees["target_label"].unique())

    ax.add_patch(Rectangle(
        (circ_floor, tox_floor), 12, 0.5,
        facecolor=PASTEL["green"], alpha=0.18, zorder=1,
    ))
    ax.axvline(circ_floor, color=DARK_GRAY,
               linestyle=":", linewidth=0.45, alpha=0.5, zorder=2)
    ax.axhline(tox_floor,  color=DARK_GRAY,
               linestyle=":", linewidth=0.45, alpha=0.5, zorder=2)

    for tid, grp in sub_fronts.groupby("topology_id"):
        target = grp["target_label"].iloc[0]
        color = S.TARGET_COLORS.get(target, "#999999")
        grp_sorted = grp.sort_values("circuit_log")
        ax.plot(grp_sorted["circuit_log"], grp_sorted["toxicity"],
                color=color, alpha=0.30, linewidth=0.6, zorder=3)

    for target in targets:
        sub_t = sub_knees[sub_knees["target_label"] == target]
        ax.scatter(
            sub_t["knee_A_circuit_log"], sub_t["knee_A_toxicity"],
            s=10, c=S.TARGET_COLORS.get(target, "#999999"),
            edgecolor="white", linewidth=0.25, zorder=5,
        )

    for yd in PAPER_YELLOW_DOTS_BY_NPN.get(npn, []):
        ax.scatter(
            yd["circuit_log"], yd["growth"],
            marker="*", s=120,
            c=ACCENT_YELLOW, edgecolor=ACCENT_YELLOW_DARK,
            linewidth=0.6, zorder=8,
        )
        ax.annotate(
            f"{yd['target']} ({yd['size']}-reg)",
            (yd["circuit_log"], yd["growth"]),
            xytext=(7, 6), textcoords="offset points",
            fontsize=5.0, color=DARK_GRAY,
            fontweight="normal", zorder=9,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=ACCENT_YELLOW_DARK,
                linewidth=0.4, alpha=0.92,
            ),
            arrowprops=dict(arrowstyle="-", color=DARK_GRAY, linewidth=0.3),
        )

    n_topo = sub_knees.shape[0]
    ax.set_title(
        f"NPN {npn}  ·  {len(targets)} target(s), n_topo={n_topo}",
        fontsize=6.0, fontweight="normal", color=TEXT_GREY,
        loc="left", pad=3,
    )

    legend_handles = [
        Line2D([], [], marker="o", color="none",
               markerfacecolor=S.TARGET_COLORS.get(t, "#999999"),
               markeredgecolor="white", markersize=4, label=t)
        for t in targets
    ]
    leg = ax.legend(
        handles=legend_handles, loc="lower left",
        frameon=False, fontsize=5.0,
        handletextpad=0.25, columnspacing=0.5,
        labelspacing=0.18, borderpad=0.05,
        ncol=2 if len(targets) > 3 else 1,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_GREY)

    ax.grid(True, linewidth=0.25, alpha=0.35,
            color=DARK_GRAY, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK_GRAY)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(
        axis="both", which="major",
        color=DARK_GRAY, width=0.6, length=2.5,
        labelsize=5.5, labelcolor=DARK_GRAY,
    )

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("circuit_log", fontsize=5.5, color=TEXT_GREY, labelpad=1.5)
    ax.set_ylabel("toxicity",   fontsize=5.5, color=TEXT_GREY, labelpad=1.5)


def build_panels() -> None:
    use_style()

    knees   = L.load_pareto_knees("G3")
    fronts  = L.load_pareto_fronts("G3")
    summary = L.load_pareto_summary("G3")

    knees["npn_class"]    = knees["source"].apply(S.npn_class_of)
    fronts["npn_class"]   = fronts["source"].apply(S.npn_class_of)
    fronts["target_label"] = fronts["source"].apply(S.parse_source)
    knees["target_label"]  = knees["source"].apply(S.parse_source)

    npn_counts = (knees.groupby("npn_class")["target_label"].nunique()
                  .sort_values(ascending=False))
    npn_classes_to_show = [c for c in S.NPN_CLASS_ORDER if c in npn_counts.index]

    circ_floor = summary["buildable_thresholds"]["circuit_log_min"]
    tox_floor  = summary["buildable_thresholds"]["toxicity_min"]

    # Shared axis limits so every per-NPN cell uses identical scales —
    # otherwise visually comparing them across loose tiles in Illustrator
    # is misleading.
    xlo = min(knees["knee_A_circuit_log"].min() - 0.3, -1.0)
    xhi = max(knees["knee_A_circuit_log"].max() + 0.5, 10.0)
    ylo, yhi = 0.65, 1.0

    print(f"Rendering {len(npn_classes_to_show)} NPN cells "
          f"at {CELL_W_MM} × {CELL_H_MM} mm each.")

    for npn in npn_classes_to_show:
        sub_fronts = fronts[fronts["npn_class"] == npn]
        sub_knees  = knees[knees["npn_class"] == npn]

        fig = plt.figure(figsize=figsize_mm(CELL_W_MM, CELL_H_MM))
        # Same proportional padding as the brick-tile cell — left margin
        # for the y-label, bottom margin for the x-label, top margin for
        # the cell title.
        ax = fig.add_axes((0.16, 0.14, 0.80, 0.74))
        _render_one(
            ax, npn=npn,
            sub_fronts=sub_fronts, sub_knees=sub_knees,
            circ_floor=circ_floor, tox_floor=tox_floor,
            xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
        )

        # Filename-safe NPN key (drop the '0x' prefix to avoid filesystem
        # quirks on case-insensitive shells; keep the canonical form in
        # the cell title above).
        safe_key = npn.replace("0x", "").lower()
        out_base = PANELS_VEC_DIR / f"panel_b_npn_{safe_key}"
        written = save_panel(fig, out_base, dpi=600, close=True)
        print(f"  NPN {npn}:")
        for fmt, path in written.items():
            print(f"    {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panels()
