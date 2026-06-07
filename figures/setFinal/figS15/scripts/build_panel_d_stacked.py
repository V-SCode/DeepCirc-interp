"""figS05 Panel D — Single-body Shapley stacked bars (REAL data).

Renders per-design fractional Shapley contributions across the 30 Panel
C designs (15 max-circuit per-target + 15 knee per-target at paper
MLP-gate thresholds), one panel per score type (circuit / growth).

Data source: data/G3/panel_c_shapley/shapley_per_design.json — produced
by topology/scripts/29_panel_c_shapley.py on cluster.

For each design d, slot k holding part p:
  fractional Φ_k = Φ_k / Σ_l |Φ_l|
where Φ_k = MLP(d) − E_{q valid at slot k}[MLP(d with q at slot k)].
Per-design fractions sum to ±1.0 (sign depends on whether the design's
parts collectively beat or lag random valid completions on that score).

Layout follows the mock variant (build_panel_d_mock_stacked.py); only
the data source changed.

Outputs:
  panels/vector/panel_d.{pdf,svg}     ← canonical figS05 Panel D
  panels/raster/panel_d.png
  panels/vector/panel_draft_d_stacked_real.{pdf,svg}  ← duplicate for QA
  panels/raster/panel_draft_d_stacked_real.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

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

CLUSTER_MAXCIRC = "#9E9E9E"   # MID_GRAY — matches S04 Panel C max-circuit dots
CLUSTER_KNEE    = "#111111"   # near-black — matches S04 Panel C pareto-optimal dots

# Top-10 colour palette — all paper-aligned (FAMILY_COLORS + PASTEL +
# 2 ACCENT_*_DARK for the sibling-RBS variants that need a distinct
# hue from their family-mate). Same idiom as the rest of paper_figures/
# — no off-palette hues. See [[feedback-paper-palette-only]].
PART_COLOURS = {
    # Paper-aligned family colours from Extended Data Fig. 4 (Option B
    # — paper-exact for single-RBS families, paper-exact for the
    # "primary" RBS of multi-RBS families, and a darker shade variant
    # for the secondary RBS so stack segments stay visually distinct
    # within the same family). BetI bumped from the paper's very pale
    # #D0E0F0 to #90B5D0 for readability in stack segments.
    "PhlF/P2":  "#F0B070",   # paper PhlF orange/peach
    "QacR/Q2":  "#E06070",   # paper QacR red/rose
    "QacR/Q1":  "#B04050",   # QacR darker variant
    "BetI/E1":  "#90B5D0",   # paper BetI (saturation-bumped pale blue)
    "AmtR/A1":  "#90C080",   # paper AmtR light green
    "PsrA/R1":  "#C0A0D0",   # paper PsrA lavender
    "SrpR/S4":  "#5080C0",   # paper SrpR medium blue
    "SrpR/S1":  "#305080",   # SrpR darker variant
    "BM3R1/B2": "#F09090",   # paper BM3R1 coral
    "BM3R1/B3": "#C06060",   # BM3R1 darker variant
}
OTHER_COLOUR = "#cccccc"

CANVAS_W_MM = 360
CANVAS_H_MM = 210


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
    """For a single design, return {part_name: fractional Φ}.

    fractional Φ_k = Φ_k / Σ_l |Φ_l|.
    Slot index k maps to part design['part_names'][k]."""
    phi_key = "phi_circuit" if score == "circuit" else "phi_growth"
    phis    = design[phi_key]
    parts   = design["part_names"]
    denom = sum(abs(v) for v in phis) or 1.0
    out: dict[str, float] = {}
    for slot, p in enumerate(parts):
        # Sum if a part appears at multiple slots (shouldn't, but
        # defensive — the no-repeated-protein constraint ensures each
        # part is at one slot at most per design).
        out[p] = out.get(p, 0.0) + phis[slot] / denom
    return out


# ---------- rendering -------------------------------------------------------

def _render_stacked(ax, designs: list[dict], score: str,
                     parts_order: list[str], score_label: str, *,
                     show_x_labels: bool = False) -> None:
    n = len(designs)
    bar_w = 0.78

    for i, d in enumerate(designs):
        fracs = _fractional_phi(d, score)
        pos_items, neg_items, other_pos, other_neg = [], [], 0.0, 0.0
        for p, v in fracs.items():
            if p in PART_COLOURS:
                (pos_items if v >= 0 else neg_items).append((p, v))
            else:
                if v >= 0:
                    other_pos += v
                else:
                    other_neg += v
        pos_items.sort(key=lambda x: -x[1])     # biggest positive at bottom
        neg_items.sort(key=lambda x:  x[1])     # biggest negative at top
        if other_pos > 0:
            pos_items.append(("other", other_pos))
        if other_neg < 0:
            neg_items.append(("other", other_neg))

        bottom = 0.0
        for p, v in pos_items:
            color = PART_COLOURS.get(p, OTHER_COLOUR)
            ax.bar(i, v, width=bar_w, bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.3,
                    zorder=3)
            bottom += v
        bottom = 0.0
        for p, v in neg_items:
            color = PART_COLOURS.get(p, OTHER_COLOUR)
            ax.bar(i, v, width=bar_w, bottom=bottom,
                    color=color, edgecolor="white", linewidth=0.3,
                    zorder=3)
            bottom += v

    # Baseline.
    ax.axhline(0, color=DARK_GRAY, linewidth=0.6, zorder=4)

    # Per-design group marker: small filled circle directly below
    # each bar. Black dot = pareto-optimal, grey dot = max circuit
    # score. Replaces the previous full-width "stripe" — at small
    # bar widths the stripes read as additional data bars and
    # confused readers; a dot is unambiguously a marker, not data.
    y_marker = -1.12
    for i, d in enumerate(designs):
        col = CLUSTER_MAXCIRC if d["group"] == "max-circ" else CLUSTER_KNEE
        ax.scatter([i], [y_marker], s=12, color=col,
                    edgecolor="none", clip_on=False, zorder=2)

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
        # Per-bar target hex (rotated 90°, sits right under group strip).
        for i, d in enumerate(designs):
            ax.text(i, -0.06, d["target"],
                      ha="center", va="top", fontsize=12,
                      color="#000000",
                      rotation=90, transform=ax.get_xaxis_transform())
        # Size-group block: a horizontal "⎵"-style grouping bracket
        # spans the hex column for each size group, with the "N-reg"
        # label sitting under the bracket. Bracket line at y_line and
        # tick arms reach UP to y_tick (toward the hex labels above).
        sizes_present = [d["gate_count"] for d in designs]
        y_tick      = -0.30          # arm tops (near hex labels)
        y_line      = -0.33          # horizontal line of the bracket
        y_label     = -0.36          # baseline for the "N-reg" text
        for size_val in sorted(set(sizes_present)):
            idxs = [i for i, s in enumerate(sizes_present) if s == size_val]
            x_left  = min(idxs) - 0.4
            x_right = max(idxs) + 0.4
            cx      = (min(idxs) + max(idxs)) / 2.0
            # Bracket as a single 4-vertex polyline: ↑─────────↑
            ax.plot([x_left, x_left, x_right, x_right],
                     [y_tick, y_line, y_line, y_tick],
                     color=DARK_GRAY, linewidth=0.6,
                     solid_joinstyle="miter",
                     transform=ax.get_xaxis_transform(),
                     clip_on=False, zorder=4)
            ax.text(cx, y_label, f"{size_val} regulators",
                      ha="center", va="top", fontsize=12,
                      color="#000000",
                      transform=ax.get_xaxis_transform())

    ax.set_xticks([])
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-1.08, 1.08)
    ax.set_ylabel(r"fractional $\Phi$", fontsize=12, color="#000000")
    ax.tick_params(axis="y", labelsize=12, colors="#000000",
                    color="#000000", width=0.5, length=2.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(0.5)
    # Sub-panel "Circuit" / "Growth" label — matches figS05 Panel E
    # row label style; bumped to 8 pt to clear the figS15 floor.
    ax.set_title(score_label, fontsize=12, fontweight="normal",
                  color="#000000", loc="left", pad=4)


def build_panel() -> None:
    use_style()

    designs = _load_designs()
    parts_order = list(PART_COLOURS.keys())

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    fig.subplots_adjust(left=0.07, right=0.83, top=0.90, bottom=0.14,
                         hspace=0.18)

    ax_c = fig.add_subplot(2, 1, 1)
    _render_stacked(ax_c, designs, "circuit", parts_order,
                     r"Circuit Score - per-design fractional $\Phi$ stack",
                     show_x_labels=False)

    ax_g = fig.add_subplot(2, 1, 2)
    _render_stacked(ax_g, designs, "growth", parts_order,
                     r"Growth Score - per-design fractional $\Phi$ stack",
                     show_x_labels=True)

    # Right-side legend — parts up top, then a "group" sub-block at
    # the bottom showing what each colour stripe under a bar means.
    lgnd_items = list(PART_COLOURS.items()) + [("other", OTHER_COLOUR)]
    legend_x  = 0.85
    legend_y0 = 0.85
    legend_dy = 0.06
    swatch_w = 0.018
    swatch_h = 0.034
    for i, (part, color) in enumerate(lgnd_items):
        y = legend_y0 - i * legend_dy
        # Rectangle vertically centred on y (text sits at y with
        # va="center", so the swatch's centre must match it).
        fig.add_artist(Rectangle((legend_x, y - swatch_h / 2),
                                    swatch_w, swatch_h,
                                    facecolor=color,
                                    edgecolor="none",
                                    transform=fig.transFigure))
        # Default sans-serif (Arial) — no monospace override so the
        # legend reads in the same Arial face as the rest of Panel D.
        fig.text(legend_x + 0.025, y, part,
                  fontsize=12, va="center", ha="left",
                  color="#000000")

    # Group-stripe legend block (moved here from the old hero subtitle).
    group_y0 = legend_y0 - len(lgnd_items) * legend_dy - 0.04
    fig.text(legend_x, group_y0 + 0.045, "Group Marker",
              fontsize=12, fontweight="bold", color="#000000",
              ha="left", va="center")
    group_items = [
        (CLUSTER_KNEE,    "Pareto-optimal"),
        (CLUSTER_MAXCIRC, "Max circuit score"),
    ]
    for i, (color, label) in enumerate(group_items):
        y = group_y0 - i * legend_dy
        # Filled circle marker centred on y (matches the under-bar
        # group marker dot used in the chart). Drawn via Circle patch
        # rather than scatter so it can be a fig.add_artist with a
        # figure-fraction transform like the other legend entries.
        marker = mpatches.Circle(
            (legend_x + swatch_w / 2, y), 0.006,
            facecolor=color, edgecolor="none",
            transform=fig.transFigure,
        )
        fig.add_artist(marker)
        fig.text(legend_x + 0.025, y, label,
                  fontsize=12, va="center", ha="left",
                  color="#000000")

    fig.text(0.50, 0.975,
              "Single-component score contribution per part per circuit design",
              ha="center", va="top",
              fontsize=16, fontweight="normal", color="#000000")

    out_base = PANELS_VEC / "panel_d"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel D (real data) written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
