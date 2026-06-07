"""figS05 Panel D v2 — Magnitude-normalised stacked bars.

Variant of build_panel_d_stacked.py that displays per-design fractional
MAGNITUDE of single-body Shapley contributions (sign-stripped). Each
design's bar sums to exactly 1.0 (positive); slice height =
|Φ_k| / Σ|Φ_l|.

Use case: "for THIS design, which slot has the biggest absolute
impact on the score?" — easier to compare composition across designs
when all bars are the same height. Trade-off vs v1: loses the sign
info that distinguishes max-circ designs (growth-hurting at every
slot, all-negative stack) from knee designs (growth-helping at every
slot, all-positive stack).

Data source: same as v1 — data/G3/panel_c_shapley/shapley_per_design
.json.

Outputs:
  panels/vector/panel_d_v2.{pdf,svg}
  panels/raster/panel_d_v2.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "figures"
                       / "setFinal" / "figS13" / "scripts"))
import build_panel_c as bc  # noqa: E402  — for TOP5_COLORS only

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    ACCENT_BLUE_DARK, ACCENT_YELLOW_DARK,
    DARK_GRAY, FAMILY_COLORS, PASTEL,
)

PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS15"
              / "panels" / "vector")

JSON_PATH = (REPO_ROOT / "data" / "topology_g3"
             / "panel_c_shapley" / "shapley_per_design.json")

TEXT_GREY = "#666666"

CLUSTER_MAXCIRC = "#7E57C2"
CLUSTER_KNEE    = "#1565C0"

# Top-10 colour palette — paper-aligned (FAMILY_COLORS + PASTEL + 2
# ACCENT_*_DARK). Mirrors v1.
PART_COLOURS = {
    "PhlF/P2":  FAMILY_COLORS["PhlF"],   # "#A3CFA3"
    "QacR/Q2":  FAMILY_COLORS["QacR"],   # "#F0BFA0"
    "BetI/E1":  PASTEL["purple"],        # "#C5A1D5"
    "AmtR/A1":  PASTEL["teal"],          # "#9FD3CF"
    "PsrA/R1":  PASTEL["red"],           # "#D77A7A"
    "SrpR/S4":  PASTEL["pink"],          # "#E7A3A3"
    "BM3R1/B2": PASTEL["blue"],          # "#9EC7E8"
    "BM3R1/B3": ACCENT_BLUE_DARK,        # "#0C7AB0" — distinct from B2
    "SrpR/S1":  PASTEL["yellow"],        # "#E7D58A"
    "QacR/Q1":  ACCENT_YELLOW_DARK,      # "#C99800" — distinct from Q2
}
OTHER_COLOUR = "#cccccc"

CANVAS_W_MM = 220
CANVAS_H_MM = 150


# ---------- data assembly ---------------------------------------------------

def _load_designs() -> list[dict]:
    """Returns a sort-stable list of designs from the Shapley JSON.

    Each design carries:
      target / topology_id / gate_count / group / part_names /
      phi_circuit / phi_growth / mlp_{circuit,growth}_score
    """
    raw = json.loads(JSON_PATH.read_text())
    designs = list(raw["designs"])
    # Stable display order: by (size, group, target hex string).
    designs.sort(key=lambda d: (d["gate_count"], d["group"], d["target"]))
    return designs


def _fractional_phi(design: dict, score: str) -> dict[str, float]:
    """For a single design, return {part_name: fractional |Φ|}.

    v2 normalisation: fractional |Φ_k| = |Φ_k| / Σ_l |Φ_l|.
    Sum across slots = 1.0 per design (all non-negative). Sign info
    is stripped; the slice size encodes only magnitude of impact."""
    phi_key = "phi_circuit" if score == "circuit" else "phi_growth"
    phis    = design[phi_key]
    parts   = design["part_names"]
    denom = sum(abs(v) for v in phis) or 1.0
    out: dict[str, float] = {}
    for slot, p in enumerate(parts):
        out[p] = out.get(p, 0.0) + abs(phis[slot]) / denom
    return out


# ---------- rendering -------------------------------------------------------

def _render_stacked(ax, designs: list[dict], score: str,
                     parts_order: list[str], score_label: str, *,
                     show_x_labels: bool = False) -> None:
    n = len(designs)
    bar_w = 0.78

    for i, d in enumerate(designs):
        fracs = _fractional_phi(d, score)
        items, other_total = [], 0.0
        for p, v in fracs.items():
            if p in PART_COLOURS:
                items.append((p, v))
            else:
                other_total += v
        items.sort(key=lambda x: -x[1])         # biggest at bottom
        if other_total > 0:
            items.append(("other", other_total))

        bottom = 0.0
        for p, v in items:
            color = PART_COLOURS.get(p, OTHER_COLOUR)
            ax.bar(i, v, width=bar_w, bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.3,
                    zorder=3)
            bottom += v

    # No signed-baseline line — all stacks are positive in v2.

    # Group strip directly beneath each bar (purple/blue).
    y_strip_top = -0.06
    y_strip_h   = 0.06
    for i, d in enumerate(designs):
        col = CLUSTER_MAXCIRC if d["group"] == "max-circ" else CLUSTER_KNEE
        ax.add_patch(Rectangle((i - bar_w / 2, y_strip_top),
                                 bar_w, y_strip_h,
                                 facecolor=col, edgecolor="none",
                                 clip_on=False, zorder=2))

    # Size separators inside the plot area.
    last_size = None
    for i, d in enumerate(designs):
        if last_size is not None and d["gate_count"] != last_size:
            ax.axvline(i - 0.5, color="#aaa", linewidth=0.7, alpha=0.6,
                        zorder=1)
        last_size = d["gate_count"]

    # Below-axis labels — only on the bottom (GROWTH) panel so the
    # shared x-grouping isn't double-printed and the panel can sit
    # tight against the one above it.
    if show_x_labels:
        # Per-bar target hex (rotated 90°, sits below group strip).
        # Group strip in v2 is at axes_y ≈ -0.06 already (data
        # y ∈ [-0.08, -0.02]) — anchor target labels just below it.
        for i, d in enumerate(designs):
            ax.text(i, -0.09, d["target"],
                      ha="center", va="top", fontsize=5.0,
                      family="monospace", color=DARK_GRAY,
                      rotation=90, transform=ax.get_xaxis_transform())
        # Size-group labels below the target hex column.
        sizes_present = [d["gate_count"] for d in designs]
        for size_val in sorted(set(sizes_present)):
            idxs = [i for i, s in enumerate(sizes_present) if s == size_val]
            cx = (min(idxs) + max(idxs)) / 2.0
            ax.text(cx, -0.40, f"{size_val}-reg",
                      ha="center", va="top", fontsize=6.5,
                      fontweight="bold", color=DARK_GRAY,
                      transform=ax.get_xaxis_transform())

    ax.set_xticks([])
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.08, 1.06)
    ax.set_ylabel("fractional |Φ|", fontsize=6.5, color=TEXT_GREY)
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

    designs = _load_designs()
    parts_order = list(PART_COLOURS.keys())

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.90, bottom=0.14,
                         hspace=0.18)

    ax_c = fig.add_subplot(2, 1, 1)
    _render_stacked(ax_c, designs, "circuit", parts_order,
                     "CIRCUIT — per-design fractional |Φ| (magnitude)",
                     show_x_labels=False)

    ax_g = fig.add_subplot(2, 1, 2)
    _render_stacked(ax_g, designs, "growth", parts_order,
                     "GROWTH — per-design fractional |Φ| (magnitude)",
                     show_x_labels=True)

    # Right-side legend.
    lgnd_items = list(PART_COLOURS.items()) + [("other", OTHER_COLOUR)]
    legend_x  = 0.85
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

    fig.text(0.50, 0.975,
              "Single-body Shapley MAGNITUDE composition per design  "
              f"(n={len(designs)}, sign-stripped, sum per bar = 1.0)",
              ha="center", va="top",
              fontsize=7.5, fontweight="bold", color="#111")
    fig.text(0.50, 0.945,
              "Group strip under each bar: "
              "purple = max-circ (n=15)  ·  blue = knee (n=15)",
              ha="center", va="top",
              fontsize=6.0, color=TEXT_GREY)

    out_base = PANELS_VEC / "panel_d_v2"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel D v2 (magnitude-normalised) written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
