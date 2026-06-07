"""Figure S05 — Panel E: pairwise epistasis heatmap (2×3 matrix grid).

Faithful paper-pipeline port of `deepcirc_sae/outputs/figures/presentation/
E2_epistasis_heatmap_v2`. Same Shapley-Taylor 2-body coefficients $\Phi_{ij}$
rendered as an l × l matrix per yellow-dot topology, in the same 2-row ×
3-col layout (Circuit / Growth × 5-reg / 6-reg / 7-reg), restyled to the
paper PASTEL palette and grey/non-bold labels.

Cells coloured on a paper-pastel diverging palette
  green = synergy   ($\Phi_{ij}$ > 0)
  red   = antagonism ($\Phi_{ij}$ < 0)
Diagonal cells hatched (self-interaction absorbed into the main effect
$\Phi_i$, shown as a right-side stripe per panel). Axis labels colored by TF
family per paper FAMILY_COLORS.

Source data: data/interp_processed/
              shapley_taylor_sim_{0x2B,0x17,0x6D}.json
              (falls back to pairwise_*.json if simulator-valued
               coefficients are absent)

Outputs:
  panels/vector/panel_e.pdf | .svg
  panels/raster/panel_e.png

Run:
    python \
        figures/setFinal/figS15/scripts/build_panel_e.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    PASTEL, FAMILY_COLORS, DARK_GRAY, MID_GRAY,
    ACCENT_BLUE,
)

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS15"
              / "panels" / "vector")
MAIN_PROC  = REPO_ROOT / "data" / "interp_processed"

TOPOLOGIES       = ["0x2B", "0x17", "0x6D"]
TOPOLOGY_DISPLAY = {"0x2B": "5 regulators", "0x17": "6 regulators", "0x6D": "7 regulators"}
TOPOLOGY_COLORS  = {
    "0x2B": "#7FA970",
    "0x17": PASTEL["purple"],
    "0x6D": PASTEL["orange"],
}
OBJECTIVES = ["circuit", "growth"]
OBJECTIVE_UNIT = {"circuit": "log(c)", "growth": "-log(g)"}
OBJECTIVE_LABEL = {
    "circuit": "Circuit (c)",
    "growth":  "Growth (g)",
}

TEXT_GREY = "#666666"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_synergy_antagonism",
    # MID_GRAY (antagonism) $\\rightarrow$ white $\\rightarrow$ ACCENT_BLUE (synergy). Softer
    # negative endpoint than DARK_GRAY (per Panel C 2026-05-29
    # follow-up) — keeps the muted-negative identity without reading
    # as near-charcoal.
    [(0.0, MID_GRAY),
     (0.5, "#FFFFFF"),
     (1.0, ACCENT_BLUE)],
)

# --- canvas ----------------------------------------------------------------

CANVAS_W_MM = 360
CANVAS_H_MM = 260


# --- data ------------------------------------------------------------------

def _load_pairwise(name: str) -> dict:
    sim_fp = MAIN_PROC / f"shapley_taylor_sim_{name}.json"
    if sim_fp.exists():
        with open(sim_fp) as f:
            d = json.load(f)
        return {
            "slot_labels": d["slot_labels"],
            "main_shapley_circuit": d["circuit"]["main_shap"],
            "main_shapley_growth":  d["growth"]["main_shap"],
            "pairwise_circuit": d["circuit"]["pair_shap"],
            "pairwise_growth":  d["growth"]["pair_shap"],
        }
    with open(MAIN_PROC / f"pairwise_{name}.json") as f:
        d = json.load(f)
    return d


def _family_of_slot_label(label: str) -> str:
    return label.split("-")[0]


def _format_dp(v: float, dp: int) -> str:
    """N-decimal-place signed value formatter. Any v that rounds to 0
    at this precision is rendered as the bare "+0" rather than
    "+0.00" / "-0.000", which removes trailing "-0.00" artefacts for
    tiny negative values."""
    if round(v, dp) == 0:
        return "+0"
    return f"{v:+.{dp}f}"


# --- per-panel render ------------------------------------------------------

def _panel_heatmap(ax, ax_main, *, name: str, obj: str, data: dict):
    pair_arr  = np.array(data[f"pairwise_{obj}"])
    main_arr  = np.array(data[f"main_shapley_{obj}"])
    slot_labels = data["slot_labels"]
    n = len(slot_labels)

    abs_max_pair = float(np.max(np.abs(pair_arr)))
    if abs_max_pair <= 0:
        abs_max_pair = 1e-6
    # Growth heatmaps use a FIXED ±0.05 normalization so each 0.01
    # step produces a consistent visible shade across all three
    # growth panels (5-/6-/7-reg). Circuit heatmaps keep the
    # per-panel dynamic abs_max_pair scale (natural range produces
    # the good blue/grey spectrum).
    if obj == "growth":
        norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)
    else:
        norm = TwoSlopeNorm(vmin=-abs_max_pair, vcenter=0.0, vmax=abs_max_pair)

    mat = pair_arr.copy()
    np.fill_diagonal(mat, np.nan)

    # 8pt floor across heatmap labels, cell values, and main-effect
    # stripe to match the new figS15 minimum.
    label_fs = 12
    cell_fs  = 10.5
    main_fs  = 10.5

    # --- pair heatmap ---
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            y = n - 1 - i
            if not np.isfinite(v):
                ax.add_patch(mpatches.Rectangle(
                    (j, y), 1, 1,
                    fc="#ECECEC", ec="white", lw=0.4, hatch="///"))
                continue
            # Growth cells: display at 3dp + quantize color to 0.001
            # so every cell that displays as "+0" lands on the same
            # exact white centre. Circuit cells: 2dp display, no
            # color quantization (natural spectrum reads well).
            v_color = round(v, 3) if obj == "growth" else v
            color = DIVERGING_CMAP(norm(v_color))
            ax.add_patch(mpatches.Rectangle(
                (j, y), 1, 1,
                fc=color, ec="white", lw=0.5))
            v_str = _format_dp(v, 3 if obj == "growth" else 2)
            ax.text(j + 0.5, y + 0.5, v_str,
                    ha="center", va="center",
                    fontsize=cell_fs, color="#222")

    # --- tick labels (slot index + family on each axis) ---
    ticks = np.arange(n) + 0.5
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Tick labels rendered in TEXT_GREY normal weight (was bold),
    # matching the Circuit (c) / Growth (g) row-label convention.
    ax.set_xticklabels(
        [f"s{i}\n{_family_of_slot_label(slot_labels[i])}" for i in range(n)],
        fontsize=label_fs, fontweight="normal",
    )
    ax.set_yticklabels(
        [f"s{i} {_family_of_slot_label(slot_labels[i])}"
         for i in range(n - 1, -1, -1)],
        fontsize=label_fs, fontweight="normal",
    )
    # All tick labels rendered in TEXT_GREY (#666666) — family colour-
    # coding dropped 2026-05-29 in favour of the locked secondary-
    # label grey used across the other S05 panels (A/B/C tick labels
    # all use TEXT_GREY too).
    for label in ax.get_xticklabels():
        label.set_color("#000000")
    for label in ax.get_yticklabels():
        label.set_color("#000000")
    ax.tick_params(length=0, pad=1.5)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- per-panel header --- normal weight TEXT_GREY (was bold)
    # so it matches the Circuit (c) / Growth (g) row labels.
    max_str = _format_dp(abs_max_pair, 3 if obj == "growth" else 2).lstrip("+")
    # Use mathtext $\Phi_{ij}$ instead of Unicode "$\Phi_{ij}$" — the
    # subscript-j character (U+2C7C) is missing from Arial and
    # renders as a white missing-glyph box.
    ax.set_title(
        rf"{TOPOLOGY_DISPLAY[name]}, $\Phi_{{ij}}$ on {OBJECTIVE_UNIT[obj]}  "
        rf"(max |$\Phi$| = {max_str})",
        loc="left", fontsize=11, color="#000000",
        fontweight="normal", pad=3,
    )

    # --- right-side main-effect stripe ---
    main_max = float(np.max(np.abs(main_arr)))
    if main_max <= 0:
        main_max = 1e-6
    # Same growth-fixed normalization as the pair heatmap.
    if obj == "growth":
        main_norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.05)
    else:
        main_norm = TwoSlopeNorm(vmin=-main_max, vcenter=0.0, vmax=main_max)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, n)
    ax_main.axis("off")
    for i in range(n):
        v = main_arr[i]
        v_color = round(v, 3) if obj == "growth" else v
        color = DIVERGING_CMAP(main_norm(v_color))
        ax_main.add_patch(mpatches.Rectangle(
            (0.05, n - 1 - i), 0.90, 1,
            fc=color, ec="white", lw=0.5))
        ax_main.text(0.50, n - 1 - i + 0.5,
                     _format_dp(v, 3 if obj == "growth" else 2),
                     ha="center", va="center",
                     fontsize=main_fs, color="#222")
    ax_main.set_title("$\Phi_i$\n(main)", loc="center",
                      fontsize=11, color="#000000", pad=2)


# --- driver ---------------------------------------------------------------

def build_panel() -> None:
    use_style()

    payloads = {n: _load_pairwise(n) for n in TOPOLOGIES}

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    outer = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.0],
        hspace=0.45, wspace=0.22,
        left=0.055, right=0.985, top=0.88, bottom=0.085,
    )

    for r, obj in enumerate(OBJECTIVES):
        for c, name in enumerate(TOPOLOGIES):
            inner = outer[r, c].subgridspec(
                1, 2, width_ratios=[6, 1], wspace=0.04,
            )
            ax      = fig.add_subplot(inner[0, 0])
            ax_main = fig.add_subplot(inner[0, 1])
            _panel_heatmap(ax, ax_main,
                           name=name, obj=obj, data=payloads[name])

    # Single-row headers in the inter-row gutters: row labels (Circuit /
    # Growth) plus the figure-wide hero title.
    header_y = {"circuit": 0.915, "growth": 0.475}
    for obj in OBJECTIVES:
        y = header_y[obj]
        # Circuit / Growth row label — normal weight TEXT_GREY at
        # 6.5 pt, matches figS05 Panel D sub-panel labels.
        fig.text(0.015, y, OBJECTIVE_LABEL[obj],
                 ha="left", va="center",
                 fontsize=12, fontweight="normal", color="#000000")

    # Bottom legend strip.
    handles = [
        mpatches.Patch(color=ACCENT_BLUE, label=r"synergy   ($\Phi_{ij}$ > 0)"),
        mpatches.Patch(color="#000000",   label=r"antagonism ($\Phi_{ij}$ < 0)"),
        mpatches.Patch(facecolor="#ECECEC", edgecolor="white",
                       hatch="///",
                       label="diagonal (self-interaction absorbed in $\Phi_i$)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center",
                     ncol=3, bbox_to_anchor=(0.5, 0.005),
                     fontsize=12, frameon=False,
                     handlelength=1.6, handleheight=0.8,
                     columnspacing=2.5)
    for txt in leg.get_texts():
        txt.set_color("#000000")

    # Hero title.
    fig.text(0.50, 0.965,
             "Pairwise epistasis heatmap on part cooperativity in "
             "Figure 3 yellow dot designs",
             fontsize=16, fontweight="normal", color="#000000",
             ha="center", va="top")

    out_base = PANELS_VEC / "panel_e"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel E written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
