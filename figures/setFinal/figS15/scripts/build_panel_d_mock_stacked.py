"""figS05 Panel D mock — Single-body Shapley STACKED BARS (mock).

Two stacked panels (circuit on top, growth below) showing the per-design
fractional Shapley composition across the 30 designs in figS04 Panel C.

For each design:
  * x-position = design index (sorted by size then group then target)
  * vertical stack = per-part fractional Shapley contributions
    (Φ_k / Σ|Φ_k|), positive above 0, negative below 0
  * coloured by part (top-10 parts get distinct colours;
    everything else lumped as 'other' grey)
Group indicator strip below each bar: purple = max-circ, blue = knee.

MOCK NOTE: Same mock_shapley function as build_panel_d_mock_heatmap —
swap once real Shapley is computed.

Outputs:
  panels/vector/panel_draft_d_stacked.{pdf,svg}
  panels/raster/panel_draft_d_stacked.png
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
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

sys.path.insert(0, str(REPO_ROOT / "figures"
                       / "setFinal" / "figS13" / "scripts"))
import build_panel_c as bc  # noqa: E402

# Same mock_shapley helper as the heatmap variant.
sys.path.insert(0, str(SCRIPT_PATH.parent))
from build_panel_d_mock_heatmap import mock_shapley  # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import DARK_GRAY, PASTEL  # noqa: E402

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS15"
              / "panels" / "vector")

TEXT_GREY = "#666666"

CLUSTER_MAXCIRC = "#7E57C2"
CLUSTER_KNEE    = "#1565C0"

SIZES   = [4, 5, 6, 7]
GROUPS  = ["max-circ", "knee"]
SCORES  = ["circuit", "growth"]

# Top-10 part palette: TOP5_COLORS + 5 muted paper hues so all 10 parts
# read as distinct in a stacked bar.
PART_COLOURS = {
    "PhlF/P2":  bc.TOP5_COLORS["PhlF/P2"],
    "QacR/Q2":  bc.TOP5_COLORS["QacR/Q2"],
    "BetI/E1":  bc.TOP5_COLORS["BetI/E1"],
    "AmtR/A1":  bc.TOP5_COLORS["AmtR/A1"],
    "PsrA/R1":  bc.TOP5_COLORS["PsrA/R1"],
    "SrpR/S4":  PASTEL["teal"],
    "BM3R1/B2": PASTEL["orange"],
    "BM3R1/B3": PASTEL["pink"],
    "SrpR/S1":  PASTEL["yellow"],
    "QacR/Q1":  PASTEL["red"],
}
OTHER_COLOUR = "#cccccc"

CANVAS_W_MM = 220
CANVAS_H_MM = 150


# ---------- DATA -------------------------------------------------------------

def _collect_designs() -> pd.DataFrame:
    maxc = bc._top15_maxcirc_per_target().copy()
    maxc["group"] = "max-circ"
    knee = bc._top15_knee_per_target().copy()
    knee["group"] = "knee"
    df = pd.concat([maxc, knee], ignore_index=True)
    # Sort for x-axis: by (size, group, target).
    df = df.sort_values(["gate_count", "group", "target"]).reset_index(drop=True)
    return df


def _design_shapleys(designs: pd.DataFrame, score: str
                     ) -> list[dict[str, float]]:
    """For each design, return {part_name: fractional Φ}.

    REAL replacement: compute slot-Shapley Φ_k from the MLP, then build
    {part_at_slot_k: Φ_k / Σ|Φ_l|}. Mock here: use per-part character +
    a per-design random offset, normalise to fractional within design.
    """
    out = []
    for _, r in designs.iterrows():
        parts = (ast.literal_eval(r["part_names"])
                 if isinstance(r["part_names"], str)
                 else list(r["part_names"]))
        size = int(r["gate_count"])
        raw = {p: mock_shapley(p, size, score) for p in parts}
        denom = sum(abs(v) for v in raw.values()) or 1.0
        out.append({p: v / denom for p, v in raw.items()})
    return out


# ---------- RENDER -----------------------------------------------------------

def _render_stacked(ax, designs: pd.DataFrame, shapleys: list[dict],
                     parts_order: list[str], score_label: str) -> None:
    n = len(designs)
    bar_w = 0.78

    # For each design, sort parts by signed Φ so positive contributions
    # stack up from 0 (largest at top) and negatives stack down.
    for i in range(n):
        d = shapleys[i]
        pos_items, neg_items = [], []
        for p in parts_order:
            v = d.get(p, 0.0)
            if v >= 0:
                pos_items.append((p, v))
            else:
                neg_items.append((p, v))
        # Sort each side by magnitude descending so big slices anchor
        # at the baseline.
        pos_items.sort(key=lambda x: -x[1])
        neg_items.sort(key=lambda x: x[1])
        # Handle 'other' (parts not in top-10) — already excluded since
        # parts_order is just top-10, but those parts may exist in the
        # design. Aggregate their share into one "other" slice.
        in_d_parts = set(d.keys())
        other_pos = sum(d[p] for p in in_d_parts
                        if p not in PART_COLOURS and d[p] >= 0)
        other_neg = sum(d[p] for p in in_d_parts
                        if p not in PART_COLOURS and d[p] < 0)
        if other_pos > 0:
            pos_items.append(("other", other_pos))
        if other_neg < 0:
            neg_items.append(("other", other_neg))

        # Draw positive stack.
        bottom = 0.0
        for p, v in pos_items:
            color = PART_COLOURS.get(p, OTHER_COLOUR)
            ax.bar(i, v, width=bar_w, bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.3,
                    zorder=3)
            bottom += v
        # Draw negative stack.
        bottom = 0.0
        for p, v in neg_items:
            color = PART_COLOURS.get(p, OTHER_COLOUR)
            ax.bar(i, v, width=bar_w, bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.3,
                    zorder=3)
            bottom += v

    # Baseline.
    ax.axhline(0, color=DARK_GRAY, linewidth=0.6, zorder=4)

    # Group indicator stripe below each bar.
    y_strip_top = -1.10
    y_strip_h   = 0.06
    for i, (_, r) in enumerate(designs.iterrows()):
        col = CLUSTER_MAXCIRC if r["group"] == "max-circ" else CLUSTER_KNEE
        ax.add_patch(Rectangle((i - bar_w / 2, y_strip_top),
                                 bar_w, y_strip_h,
                                 facecolor=col, edgecolor="none",
                                 clip_on=False, zorder=2))

    # Size separators on x-axis (between size groups).
    last_size = None
    for i, (_, r) in enumerate(designs.iterrows()):
        if last_size is not None and r["gate_count"] != last_size:
            ax.axvline(i - 0.5, color="#aaa", linewidth=0.7, alpha=0.6,
                        zorder=1)
        last_size = r["gate_count"]
    # Size labels.
    sizes_present = designs["gate_count"].tolist()
    for size_val in sorted(set(sizes_present)):
        idxs = [i for i, s in enumerate(sizes_present) if s == size_val]
        cx = (min(idxs) + max(idxs)) / 2.0
        ax.text(cx, -1.32, f"{size_val}-reg",
                  ha="center", va="bottom", fontsize=6.5,
                  fontweight="bold", color=DARK_GRAY,
                  transform=ax.get_xaxis_transform())

    ax.set_xticks([])
    ax.set_xlim(-0.7, n - 0.3)
    ymax = max(0.9, max((sum(v for v in d.values() if v > 0)
                          for d in shapleys), default=0.5) * 1.05)
    ymin = min(-0.9, min((sum(v for v in d.values() if v < 0)
                          for d in shapleys), default=-0.5) * 1.05)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("fractional Φ", fontsize=6.5, color=TEXT_GREY)
    ax.tick_params(axis="y", labelsize=5.5, colors=DARK_GRAY,
                    color=DARK_GRAY, width=0.5, length=2.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK_GRAY)
        ax.spines[spine].set_linewidth(0.5)
    ax.set_title(score_label, fontsize=7.0, fontweight="bold",
                  color="#111", loc="left", pad=4)


def build_panel() -> None:
    use_style()

    designs = _collect_designs()
    shap_circuit = _design_shapleys(designs, "circuit")
    shap_growth  = _design_shapleys(designs, "growth")

    parts_order = list(PART_COLOURS.keys())

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.92, bottom=0.10,
                         hspace=0.55)

    ax_c = fig.add_subplot(2, 1, 1)
    _render_stacked(ax_c, designs, shap_circuit, parts_order,
                     "CIRCUIT — per-design fractional Φ stack")

    ax_g = fig.add_subplot(2, 1, 2)
    _render_stacked(ax_g, designs, shap_growth, parts_order,
                     "GROWTH — per-design fractional Φ stack")

    # Legend on the right side.
    lgnd_items = list(PART_COLOURS.items()) + [("other", OTHER_COLOUR)]
    legend_x = 0.85
    legend_y0 = 0.85
    legend_dy = 0.06
    for i, (part, color) in enumerate(lgnd_items):
        y = legend_y0 - i * legend_dy
        fig.add_artist(Rectangle((legend_x, y - 0.02), 0.018, 0.034,
                                    facecolor=color,
                                    edgecolor="none",
                                    transform=fig.transFigure))
        fig.text(legend_x + 0.025, y, part,
                  fontsize=6.0, va="center", ha="left",
                  family="monospace", color=DARK_GRAY)

    # Group-strip legend.
    fig.text(0.50, 0.025,
              "Group strip below each bar:  "
              "purple = max-circ (15 designs)     "
              "blue = knee (15 designs)",
              ha="center", va="center",
              fontsize=6.0, color=TEXT_GREY)
    fig.text(0.50, 0.97,
              "Per-design single-body Shapley composition  "
              "(n=30 designs, sorted by size & group, MOCK DATA)",
              ha="center", va="top",
              fontsize=7.5, fontweight="bold", color="#111")

    out_base = PANELS_VEC / "panel_draft_d_stacked"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("panel_draft_d_stacked written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
