"""Figure S05 — Panel C v2: blue→red palette variant.

Same data + layout as build_panel_c.py; diverging cmap swapped from
MID_GRAY → white → ACCENT_BLUE to BAD_DEEP red → white → ACCENT_BLUE,
and per-row tick highlights for DESTRUCTIVE_EX / DESTRUCTIVE_T2 also
swapped to the matching red tiers (matches Panels A/B v2).

Outputs: panels/vector/panel_c_v2.{pdf,svg}
         panels/raster/panel_c_v2.png

Run:
    python \\
        figures/setFinal/figS14/scripts/build_panel_c_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
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
    DARK_GRAY, MID_GRAY, LIGHT_GRAY,
    ACCENT_BLUE, ACCENT_BLUE_DARK,
)

sys.path.insert(0, str(SCRIPT_PATH.parent))
from _palette_v2 import BAD_DEEP, BAD_TIER2, BAD_CMAP_END  # noqa: E402


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS14"
              / "panels" / "vector")

TEXT_GREY = "#666666"

BUCKETS = [
    "NOT-early",
    "NOT-middle",
    "NOT-late",
    "NOR-early",
    "NOR-middle",
    "NOR-late",
]

BUCKET_LABEL = {
    "NOT-early":  "NOT\nearly",
    "NOT-middle": "NOT\nmiddle",
    "NOT-late":   "NOT\nlate",
    "NOR-early":  "NOR\nearly",
    "NOR-middle": "NOR\nmiddle",
    "NOR-late":   "NOR\nlate",
}


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


def make_red_blue_diverging() -> mcolors.LinearSegmentedColormap:
    """BAD_CMAP_END (deep red) → white → ACCENT_BLUE (cyan).

    v2 variant of the grey/blue diverging cmap — negatives are red
    instead of grey so "bad cells" read chromatically.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "paper_red_blue_div_v2",
        [
            (0.00, BAD_CMAP_END),
            (0.50, "#FFFFFF"),
            (1.00, ACCENT_BLUE),
        ],
        N=256,
    )


CANVAS_W_MM = 174
CANVAS_H_MM = 120


def build_panel() -> None:
    use_style()

    l2 = pd.read_csv(L2_TOP05_CSV)
    elig = l2[
        (l2["task"] == "circuit_log")
        & (l2["n_topologies"] >= 5)
        & (l2["n_targets"] >= 3)
        & (l2["total_designs"] >= 1000)
    ].copy()

    elig["bucket"] = elig.apply(bucket_of, axis=1)

    parts_by_median = (
        elig.groupby("part_name")["log_odds"]
        .median()
        .sort_values()
        .index.tolist()
    )

    grid = (
        elig.groupby(["part_name", "bucket"])["log_odds"]
        .mean()
        .unstack("bucket")
        .reindex(index=parts_by_median, columns=BUCKETS)
    )

    finite = grid.values[np.isfinite(grid.values)]
    abs_p98 = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(abs_p98, 1.0)
    vmin = -vmax
    cmap = make_red_blue_diverging()

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes((0.14, 0.08, 0.72, 0.78))
    cax = fig.add_axes((0.89, 0.08, 0.018, 0.78))

    A = grid.values
    mask = np.isnan(A)
    Adisp = np.where(mask, 0.0, A)
    im = ax.imshow(Adisp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                    interpolation="nearest")

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor=LIGHT_GRAY,
                                            edgecolor="none", zorder=2))
                ax.text(j, i, "—", color=TEXT_GREY, fontsize=5.5,
                         ha="center", va="center", zorder=3)
            else:
                v = A[i, j]
                txt_color = "white" if abs(v) > 0.55 * vmax else "#333333"
                ax.text(j, i, f"{v:+.2f}", color=txt_color, fontsize=5.5,
                         ha="center", va="center", zorder=3)

    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKETS],
                        fontsize=5.5, color=TEXT_GREY)
    ax.set_yticks(range(len(parts_by_median)))
    ax.set_yticklabels(parts_by_median, fontsize=5.5, color=TEXT_GREY)

    CONSTRUCTIVE = {"PhlF/P2", "PhlF/P3", "SrpR/S2", "SrpR/S4", "AmtR/A1"}
    DESTRUCTIVE_T2 = {"SrpR/S1", "HlyIIR/H1", "LitR/L1", "LmrA/N1"}
    DESTRUCTIVE_EX = {"PhlF/P1"}
    for tick, part in zip(ax.get_yticklabels(), parts_by_median):
        if part in DESTRUCTIVE_EX:
            tick.set_color(BAD_DEEP)
        elif part in DESTRUCTIVE_T2:
            tick.set_color(BAD_TIER2)
        elif part in CONSTRUCTIVE:
            tick.set_color(ACCENT_BLUE_DARK)

    ax.set_xlim(-0.5, len(BUCKETS) - 0.5)
    ax.set_ylim(len(parts_by_median) - 0.5, -0.5)
    ax.tick_params(axis="x", color=DARK_GRAY, width=0.6, length=2.0,
                    pad=2.0)
    ax.tick_params(axis="y", color=DARK_GRAY, width=0.6, length=2.0)

    for i in range(len(parts_by_median) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.4, zorder=4)
    for j in range(len(BUCKETS) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.4, zorder=4)

    ax.axvline(2.5, color=DARK_GRAY, linewidth=0.6, zorder=5)

    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=4.8, color=DARK_GRAY, width=0.6,
                       length=2.0, labelcolor=TEXT_GREY)
    cb.set_label(r"mean log$_2$ enrichment (top-5% / bot-5% by Circuit Score (log))",
                  fontsize=5.5, color=TEXT_GREY, labelpad=4.0)

    ax.set_title(
        r"Mean circuit score log$_2$ enrichment per part across circuit positions",
        fontsize=10, fontweight="normal", color=DARK_GRAY,
        loc="center", pad=6)

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    out_base = PANELS_VEC / "panel_c_v2"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel C v2 written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
