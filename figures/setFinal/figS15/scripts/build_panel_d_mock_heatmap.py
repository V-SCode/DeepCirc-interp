"""figS05 Panel D mock — Single-body Shapley HEATMAP (mock).

Two side-by-side heatmaps (circuit | growth) summarising per-part
single-body Shapley contributions across the 30 designs in figS04
Panel C (15 max-circ per-target + 15 knee per-target).

Rows = top-10 most-frequent parts in the 30-design set.
Cols = 4 sizes × 2 groups (4-reg M | 4-reg K | 5-reg M | 5-reg K | …).
Cell = mean fractional Shapley (Φ_k / Σ|Φ_k|) across all designs in
       that (size, group) bin where the part appears at some slot.
Cells with no designs in that bin are rendered N/A (light grey + slash).

MOCK NOTE: Shapley values here are synthesised from a per-part character
matrix + light noise. Same plot script will be re-used once real Shapley
runs against the MLPs on cluster — only the `mock_shapley` function
needs to be swapped for the real one.

Outputs:
  panels/vector/panel_draft_d_heatmap.{pdf,svg}
  panels/raster/panel_draft_d_heatmap.png
"""
from __future__ import annotations

import ast
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse Panel C helpers — same 30-design source.
sys.path.insert(0, str(REPO_ROOT / "figures"
                       / "setFinal" / "figS13" / "scripts"))
import build_panel_c as bc  # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import DARK_GRAY, LIGHT_GRAY  # noqa: E402

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS15"
              / "panels" / "vector")

TEXT_GREY = "#666666"
NA_COLOR  = "#E8E8E8"   # N/A cells

CLUSTER_MAXCIRC = "#7E57C2"   # purple (matches Panel C)
CLUSTER_KNEE    = "#1565C0"   # blue   (matches Panel C)

SIZES   = [4, 5, 6, 7]
GROUPS  = ["max-circ", "knee"]
SCORES  = ["circuit", "growth"]

CANVAS_W_MM = 220
CANVAS_H_MM = 150

RNG = np.random.default_rng(42)


# ---------- MOCK SHAPLEY -----------------------------------------------------

# Per-part character matrix: (part, score) -> centre Shapley value.
# Designed so a few stories are visible:
#   * PhlF/P2 ~ mildly + circuit, ~ neutral growth (universal)
#   * QacR/Q2 ~ neutral circuit, strongly + growth (protective)
#   * PsrA/R1 ~ strongly + circuit, strongly − growth (max-circ favourite)
#   * IcaRA/I1 ~ + circuit, very − growth (toxic, max-circ-only)
PART_CHAR = {
    ("PhlF/P2",  "circuit"): +0.18,  ("PhlF/P2",  "growth"): +0.04,
    ("QacR/Q2",  "circuit"): -0.02,  ("QacR/Q2",  "growth"): +0.20,
    ("BetI/E1",  "circuit"): +0.08,  ("BetI/E1",  "growth"): +0.05,
    ("AmtR/A1",  "circuit"): +0.10,  ("AmtR/A1",  "growth"): +0.02,
    ("SrpR/S4",  "circuit"): +0.04,  ("SrpR/S4",  "growth"): -0.02,
    ("PsrA/R1",  "circuit"): +0.25,  ("PsrA/R1",  "growth"): -0.15,
    ("BM3R1/B2", "circuit"): +0.04,  ("BM3R1/B2", "growth"): -0.02,
    ("BM3R1/B3", "circuit"): +0.02,  ("BM3R1/B3", "growth"): -0.01,
    ("SrpR/S1",  "circuit"): +0.06,  ("SrpR/S1",  "growth"): -0.04,
    ("QacR/Q1",  "circuit"): +0.18,  ("QacR/Q1",  "growth"): -0.25,
    ("IcaRA/I1", "circuit"): +0.15,  ("IcaRA/I1", "growth"): -0.30,
    ("BM3R1/B1", "circuit"): +0.02,  ("BM3R1/B1", "growth"): -0.05,
}


def mock_shapley(part: str, size: int, score: str) -> float:
    """Return a mock fractional Shapley value for (part, size, score).

    REAL replacement: compute single-body Shapley on the MLP for the
    specific design, normalise by Σ|Φ|, then average across designs in
    the bin. Same return shape — drop-in.
    """
    base = PART_CHAR.get((part, score), 0.0)
    size_factor = {4: 1.30, 5: 1.10, 6: 0.90, 7: 0.75}.get(size, 1.0)
    noise = RNG.normal(0.0, 0.02)
    return float(base * size_factor + noise)


# ---------- DATA ASSEMBLY ----------------------------------------------------

def _collect_designs() -> pd.DataFrame:
    """Concatenate the 30 Panel C designs with a `group` column."""
    maxc = bc._top15_maxcirc_per_target().copy()
    maxc["group"] = "max-circ"
    knee = bc._top15_knee_per_target().copy()
    knee["group"] = "knee"
    return pd.concat([maxc, knee], ignore_index=True)


def _top10_parts(designs: pd.DataFrame) -> list[str]:
    """Top-10 parts by total presence across the 30 designs."""
    counter: Counter = Counter()
    for parts_str in designs["part_names"]:
        parts = (ast.literal_eval(parts_str)
                 if isinstance(parts_str, str) else list(parts_str))
        counter.update(parts)
    return [p for p, _ in counter.most_common(10)]


def _build_heatmap(designs: pd.DataFrame, parts: list[str],
                    score: str
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix, mask) for a single score-type heatmap.

    matrix shape: (n_parts, n_size * n_group). Mask is True for cells
    that have NO designs in the bin (rendered as N/A)."""
    cols = [(s, g) for s in SIZES for g in GROUPS]
    n_rows, n_cols = len(parts), len(cols)
    mat  = np.zeros((n_rows, n_cols))
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    for ci, (size, group) in enumerate(cols):
        bin_designs = designs[(designs["gate_count"] == size)
                              & (designs["group"] == group)]
        if len(bin_designs) == 0:
            mask[:, ci] = True
            continue
        for ri, part in enumerate(parts):
            # Find designs in this bin that contain `part`.
            present = []
            for _, r in bin_designs.iterrows():
                row_parts = (ast.literal_eval(r["part_names"])
                              if isinstance(r["part_names"], str)
                              else list(r["part_names"]))
                if part in row_parts:
                    present.append(r)
            if not present:
                mask[ri, ci] = True
                continue
            # Average mock Shapley across present designs.
            vals = [mock_shapley(part, size, score) for _ in present]
            mat[ri, ci] = float(np.mean(vals))
    return mat, mask


# ---------- RENDER -----------------------------------------------------------

def _render_one_heatmap(ax, mat, mask, parts, score_label, vmax):
    cols = [(s, g) for s in SIZES for g in GROUPS]
    n_rows, n_cols = mat.shape

    # Background N/A fill first.
    for ri in range(n_rows):
        for ci in range(n_cols):
            if mask[ri, ci]:
                ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                         facecolor=NA_COLOR,
                                         edgecolor="white", linewidth=0.4,
                                         zorder=1))
    masked = np.ma.array(mat, mask=mask)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    aspect="auto", origin="upper", zorder=2)

    # Annotate each cell with the numeric value (skip N/A).
    for ri in range(n_rows):
        for ci in range(n_cols):
            if mask[ri, ci]:
                ax.text(ci, ri, "N/A", ha="center", va="center",
                          fontsize=5.0, color="#888", style="italic")
            else:
                v = mat[ri, ci]
                txt_color = "#fff" if abs(v) > vmax * 0.55 else "#222"
                ax.text(ci, ri, f"{v:+.02f}", ha="center", va="center",
                          fontsize=5.0, color=txt_color)

    ax.set_xticks(range(n_cols))
    # Two-level x labels: size on row 0, group on row 1.
    xticklabels = [f"{g[0].upper()}" for _, g in cols]   # M / K
    ax.set_xticklabels(xticklabels, fontsize=5.5, color=DARK_GRAY)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(parts, fontsize=6.0, family="monospace",
                       color=DARK_GRAY)

    # Group-pair separators every 2 cols + size annotation above.
    for i in range(0, n_cols, 2):
        if i > 0:
            ax.axvline(i - 0.5, color="#aaa", linewidth=0.7)
        size = cols[i][0]
        ax.text(i + 0.5, -0.85, f"{size}-reg",
                  ha="center", va="bottom", fontsize=6.0,
                  fontweight="bold", color=DARK_GRAY)
    # Group colour stripe under each col header.
    for ci, (_, g) in enumerate(cols):
        color = CLUSTER_MAXCIRC if g == "max-circ" else CLUSTER_KNEE
        ax.add_patch(Rectangle((ci - 0.5, -0.35), 1, 0.18,
                                 facecolor=color, edgecolor="none",
                                 clip_on=False, zorder=3))

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -1.10)
    ax.set_title(score_label, fontsize=7.0, fontweight="bold",
                  color="#111", loc="left", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both",
                    length=0, color=DARK_GRAY)
    return im


def build_panel() -> None:
    use_style()

    designs = _collect_designs()
    parts = _top10_parts(designs)

    mats = {sc: _build_heatmap(designs, parts, sc) for sc in SCORES}
    # Single shared vmax for symmetric diverging cmap.
    vmax = max(abs(mats[sc][0]).max() for sc in SCORES)
    vmax = max(vmax, 0.10)

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    fig.subplots_adjust(left=0.10, right=0.94, top=0.86, bottom=0.13,
                         wspace=0.18)

    ax_c = fig.add_subplot(1, 2, 1)
    im_c = _render_one_heatmap(ax_c, *mats["circuit"], parts,
                                 "CIRCUIT — mean fractional Φ per (size, group)",
                                 vmax)
    ax_g = fig.add_subplot(1, 2, 2)
    im_g = _render_one_heatmap(ax_g, *mats["growth"], parts,
                                 "GROWTH — mean fractional Φ per (size, group)",
                                 vmax)
    # Hide y-tick labels on the right heatmap (shared parts axis).
    ax_g.set_yticklabels([])

    # Shared colorbar.
    cbar_ax = fig.add_axes([0.95, 0.20, 0.012, 0.55])
    cb = fig.colorbar(im_g, cax=cbar_ax)
    cb.set_label("mean fractional Φ", fontsize=6.0, color=TEXT_GREY)
    cb.ax.tick_params(labelsize=5.5, color=DARK_GRAY)
    cb.outline.set_visible(False)

    # Legend strip for the M/K colour stripes.
    fig.text(0.50, 0.04,
              "M = max-circ group (purple)     "
              "K = knee group (blue)     "
              "N/A = no designs in that (size, group)",
              ha="center", va="center",
              fontsize=6.0, color=TEXT_GREY)

    fig.text(0.50, 0.97,
              "Single-body Shapley by part × circuit size × selection group "
              "(n=30 designs, MOCK DATA)",
              ha="center", va="top",
              fontsize=7.5, fontweight="bold", color="#111")

    out_base = PANELS_VEC / "panel_draft_d_heatmap"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("panel_draft_d_heatmap written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
