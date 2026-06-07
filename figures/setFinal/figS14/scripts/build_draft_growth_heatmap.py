"""DRAFT panel — parts × role-bucket GROWTH log-odds heatmap.

Growth-score analog of `build_panel_c.py` (which renders the same
6-bucket position-aware view but for `task=circuit_log`). Sandbox
panel; NOT placed in figS05's manifest.

Pairs with Panel A (growth beeswarm) the same way Panel C pairs with
Panel B (circuit beeswarm). Together with Panel C you can read off
whether a part's growth/circuit effect is **role-uniform** (same color
across all 6 buckets — part-level rule) or **role-specific** (sign
flips along the cascade — position-level rule).

Layout:
  rows = 20 library parts sorted by overall median toxicity log_odds
         (most-toxic top; matches the growth beeswarm in Panel A).
  cols = 6 role buckets, ordered left-to-right along the signal cascade:
         {NOT, NOR} × {early (d2o>=3, input-adjacent),
                       middle (d2o=2),
                       late (d2o=1, output-adjacent)}
  cell = mean log_odds across universal-eligible fp_keys in that
         (part, bucket). Empty cells drawn as light-gray with "—".
  cbar = PASTEL green↔red diverging, symmetric around 0.

Source: data/topology_g3/l2_top05/l2_enrichment.csv (task=toxicity)

Outputs:
  panels/vector/panel_draft_growth_heatmap.pdf | .svg
  panels/raster/panel_draft_growth_heatmap.png

Run:
    python \\
        figures/setFinal/figS14/scripts/\\
        build_draft_growth_heatmap.py
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
    PASTEL, DARK_GRAY, MID_GRAY, LIGHT_GRAY,
)

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


def make_pastel_diverging() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "pastel_div",
        [
            (0.00, PASTEL["red"]),
            (0.50, "#FFFFFF"),
            (1.00, PASTEL["green"]),
        ],
        N=256,
    )


CANVAS_W_MM = 174
CANVAS_H_MM = 120


def build_panel() -> None:
    use_style()

    l2 = pd.read_csv(L2_TOP05_CSV)
    elig = l2[
        (l2["task"] == "toxicity")
        & (l2["n_topologies"] >= 5)
        & (l2["n_targets"] >= 3)
        & (l2["total_designs"] >= 1000)
    ].copy()

    elig["bucket"] = elig.apply(bucket_of, axis=1)

    # Sort parts by overall median (matches Panel A's growth beeswarm).
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
    counts = (
        elig.groupby(["part_name", "bucket"])["log_odds"]
        .size()
        .unstack("bucket")
        .reindex(index=parts_by_median, columns=BUCKETS)
        .fillna(0)
        .astype(int)
    )

    finite = grid.values[np.isfinite(grid.values)]
    abs_p98 = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(abs_p98, 1.0)
    vmin = -vmax
    cmap = make_pastel_diverging()

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
                ax.text(j, i, "—", color=TEXT_GREY, fontsize=4.5,
                         ha="center", va="center", zorder=3)
            else:
                v = A[i, j]
                txt_color = "white" if abs(v) > 0.55 * vmax else "#333333"
                ax.text(j, i, f"{v:+.2f}", color=txt_color, fontsize=4.0,
                         ha="center", va="center", zorder=3)

    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKETS],
                        fontsize=5.0, color=TEXT_GREY)
    ax.set_yticks(range(len(parts_by_median)))
    ax.set_yticklabels(parts_by_median, fontsize=5.0, color=TEXT_GREY)

    # Row tick color = same toxic/neutral/protective scheme as Panel A.
    TOXIC_EXTREME = {"IcaRA/I1"}
    TOXIC_T2 = {"QacR/Q1", "HlyIIR/H1"}
    PROTECTIVE = {"QacR/Q2", "PhlF/P1"}
    for tick, part in zip(ax.get_yticklabels(), parts_by_median):
        if part in TOXIC_EXTREME:
            tick.set_color(PASTEL["red"])
        elif part in TOXIC_T2:
            tick.set_color(PASTEL["orange"])
        elif part in PROTECTIVE:
            tick.set_color(PASTEL["green"])

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
    cb.set_label("mean log-odds(top-5% / bot-5% by growth_score)",
                  fontsize=5.0, color=TEXT_GREY, labelpad=4.0)

    n_cells = int((~grid.isna()).values.sum())
    bucket_counts = ", ".join(
        f"{b}={int(counts[b].max())}" for b in BUCKETS
    )
    ax.set_title(
        f"DRAFT — mean growth log-odds per part × role bucket\n"
        f"{n_cells}/120 (part × bucket) cells populated · "
        f"fp_keys per bucket: {bucket_counts}",
        fontsize=5.8, fontweight="normal", color=TEXT_GREY,
        loc="left", pad=8)

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    out_base = PANELS_VEC / "panel_draft_growth_heatmap"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Draft growth heatmap written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
