"""Figure S05 — Panel C: parts × role-bucket circuit log-odds heatmap.

Position-aware companion to Panel B. Collapses the 65 universal-eligible
fp_keys into 6 coarse role buckets defined by
(node_type × depth-to-output coarseness), and reports the **mean
log_odds** of each part within each bucket.
(Predecessor: the original Panel C was the IcaRA/I1 log-odds density
view ported from `fig02_toxic_class`; preserved at
`scripts/_archive/build_panel_c_icara_density.py`.)

Why this view: Panel B's beeswarm answers "is part X enriched in top-5%
on average?" — this heatmap answers "WHERE (which positions) does part
X get used in top-5% vs bot-5%?". Tells you the spatial assignment
preference the agent learned for each part.

Layout:
  rows = 20 library parts sorted by overall median circuit log_odds
         (most-destructive top; matches Panel B's y-axis ordering).
  cols = 6 role buckets, ordered left-to-right along the signal cascade:
         {NOT, NOR} × {early (d2o>=3, input-adjacent),
                       middle (d2o=2),
                       late (d2o=1, output-adjacent)}
  cell = mean log_odds across universal-eligible fp_keys in that
         (part, bucket). Empty cells (no fp_keys in that bucket pass
         the universal filter) drawn as light-gray with "—".
  cbar = PASTEL green↔red diverging, symmetric around 0.

Source: data/topology_g3/l2_top05/l2_enrichment.csv

Outputs:
  panels/vector/panel_c.pdf | .svg
  panels/raster/panel_c.png

Run:
    python \\
        figures/setFinal/figS14/scripts/build_panel_c.py
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
    ACCENT_BLUE, ACCENT_BLUE_DARK,
)

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS14"
              / "panels" / "vector")

TEXT_GREY = "#666666"

# --- role-bucket definitions ------------------------------------------------

# Order = left-to-right along the signal cascade: input-adjacent first,
# output-adjacent last. So "early" sits leftmost (gates that fire first
# as the signal propagates from sensors to YFP) and "late" sits rightmost.
BUCKETS = [
    "NOT-early",   # type=NOT, d2o>=3 (input-adjacent; 9 fp_keys)
    "NOT-middle",  # type=NOT, d2o=2  (5 fp_keys)
    "NOT-late",    # type=NOT, d2o=1  (output-adjacent; 2 fp_keys)
    "NOR-early",   # type=NOR, d2o>=3 (27 fp_keys)
    "NOR-middle",  # type=NOR, d2o=2  (11 fp_keys)
    "NOR-late",    # type=NOR, d2o=1  (11 fp_keys)
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
    """Map a graph-role fingerprint row to one of the 6 buckets, where
    'late' means d2o=1 (output-adjacent — the gate fires last) and
    'early' means d2o>=3 (input-adjacent — fires first in the cascade).
    """
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


# --- diverging PASTEL colormap ---------------------------------------------

def make_pastel_diverging() -> mcolors.LinearSegmentedColormap:
    """MID_GRAY $\\rightarrow$ light $\\rightarrow$ ACCENT_BLUE, asymmetric around 0.

    Paper-aligned grey-↔-blue diverging cmap: ACCENT_BLUE pops for
    positive log-odds while negatives go quiet on the paper's
    MID_GRAY (was DARK_GRAY — softened 2026-05-29 because the lowest
    cells were reading as near-charcoal/harsh; MID_GRAY keeps the
    "muted negative" signal without the heaviness).
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "paper_grey_blue_div",
        [
            (0.00, MID_GRAY),
            (0.50, "#FFFFFF"),
            (1.00, ACCENT_BLUE),
        ],
        N=256,
    )


# --- canvas ----------------------------------------------------------------

CANVAS_W_MM = 410
CANVAS_H_MM = 200


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

    # Sort parts by overall median (same order as the beeswarm draft).
    parts_by_median = (
        elig.groupby("part_name")["log_odds"]
        .median()
        .sort_values()
        .index.tolist()
    )

    # Aggregate: mean log_odds per (part, bucket).
    grid = (
        elig.groupby(["part_name", "bucket"])["log_odds"]
        .mean()
        .unstack("bucket")
        .reindex(index=parts_by_median, columns=BUCKETS)
    )
    # Count of cells contributing per (part, bucket) — for QA in title.
    counts = (
        elig.groupby(["part_name", "bucket"])["log_odds"]
        .size()
        .unstack("bucket")
        .reindex(index=parts_by_median, columns=BUCKETS)
        .fillna(0)
        .astype(int)
    )

    # --- color scale: symmetric around 0, robust to outliers -----------
    finite = grid.values[np.isfinite(grid.values)]
    abs_p98 = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(abs_p98, 1.0)  # don't go below ±1 so colors look ordinal
    vmin = -vmax
    cmap = make_pastel_diverging()

    # --- figure ---------------------------------------------------------
    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes((0.14, 0.08, 0.72, 0.78))
    cax = fig.add_axes((0.89, 0.08, 0.018, 0.78))

    A = grid.values
    mask = np.isnan(A)
    Adisp = np.where(mask, 0.0, A)
    im = ax.imshow(Adisp, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                    interpolation="nearest")

    # Draw empty cells as LIGHT_GRAY overlay with an em-dash.
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor=LIGHT_GRAY,
                                            edgecolor="none", zorder=2))
                ax.text(j, i, "—", color="#000000", fontsize=12,
                         ha="center", va="center", zorder=3)
            else:
                # Cell value label.
                v = A[i, j]
                txt_color = "white" if abs(v) > 0.55 * vmax else "#333333"
                ax.text(j, i, f"{v:+.2f}", color=txt_color, fontsize=12,
                         ha="center", va="center", zorder=3)

    # --- axes -----------------------------------------------------------
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([BUCKET_LABEL[b] for b in BUCKETS],
                        fontsize=12, color="#000000")
    ax.set_yticks(range(len(parts_by_median)))
    ax.set_yticklabels(parts_by_median, fontsize=12, color="#000000")

    # Per-row tick color: highlight constructive/destructive parts (same
    # palette as the beeswarm draft).
    CONSTRUCTIVE = {"PhlF/P2", "PhlF/P3", "SrpR/S2", "SrpR/S4", "AmtR/A1"}
    DESTRUCTIVE_T2 = {"SrpR/S1", "HlyIIR/H1", "LitR/L1", "LmrA/N1"}
    DESTRUCTIVE_EX = {"PhlF/P1"}
    for tick, part in zip(ax.get_yticklabels(), parts_by_median):
        if part in DESTRUCTIVE_EX:
            tick.set_color("#111111")   # near-black, matches Panels A/B
        elif part in DESTRUCTIVE_T2:
            tick.set_color(MID_GRAY)    # second-tier grey, matches Panels A/B
        elif part in CONSTRUCTIVE:
            tick.set_color(ACCENT_BLUE_DARK)

    ax.set_xlim(-0.5, len(BUCKETS) - 0.5)
    ax.set_ylim(len(parts_by_median) - 0.5, -0.5)
    ax.tick_params(axis="x", color="#000000", width=0.6, length=2.0,
                    pad=2.0)
    ax.tick_params(axis="y", color="#000000", width=0.6, length=2.0)

    # Light grid between cells.
    for i in range(len(parts_by_median) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.4, zorder=4)
    for j in range(len(BUCKETS) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.4, zorder=4)

    # Divider between NOT and NOR halves.
    ax.axvline(2.5, color=DARK_GRAY, linewidth=0.6, zorder=5)

    # --- colorbar -------------------------------------------------------
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=12, color=DARK_GRAY, width=0.6,
                       length=2.0, labelcolor="#000000")
    cb.set_label(r"mean log$_2$ enrichment (top-5% / bot-5% by Circuit Score (log))",
                  fontsize=12, color="#000000", labelpad=4.0)

    # --- title ----------------------------------------------------------
    # Reverted from fig.text() back to ax.set_title() so the title sits
    # tight against the top of the heatmap (the fig.text version at
    # y=0.975 created a visible white gap above the axes in the
    # composed figure).
    ax.set_title(
        r"Mean circuit score log$_2$ enrichment per part across circuit positions",
        fontsize=19, fontweight="normal", color="#000000",
        loc="center", pad=6)

    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    out_base = PANELS_VEC / "panel_c"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel C written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
