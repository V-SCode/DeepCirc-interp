"""figS04 Panel C — top-5 highlight test variants.

Builds two test renderings of Panel C in which the top-5 most-frequent
parts across the per-target best designs are visually distinguished
inside the right-side table, so a reader can scan rows to see WHERE
each universal part appears (and which positions it occupies):

  v2 — coloured TEXT for top-5 parts; non-top-5 parts stay grey.
  v3 — coloured BOX OUTLINE around top-5 parts; all text stays grey.

Top-5 part frequencies in the n=20 per-target best set (paper MLP gate):
  PhlF/P2 20/20   PsrA/R1 19/20   AmtR/A1 16/20
  BetI/E1 14/20   QacR/Q2 13/20

Outputs:
  panels/vector/panel_c_v2.{pdf,svg}
  panels/raster/panel_c_v2.png
  panels/vector/panel_c_v3.{pdf,svg}
  panels/raster/panel_c_v3.png
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

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
    ACCENT_YELLOW, ACCENT_YELLOW_DARK, DARK_GRAY, MID_GRAY, PASTEL,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS13"
              / "panels" / "vector")

TEXT_GREY = "#666666"
PASS_GREEN = "#1A9850"
FAIL_RED   = "#D73027"

# Paper MLP-gate thresholds.
CIRC_SCORE_FLOOR = 2.0
CIRC_LOG_FLOOR   = float(np.log(CIRC_SCORE_FLOOR))
TOX_FLOOR        = 0.5

# Top-5 parts in the per-target best set, with distinct paper-consistent
# hues. Ordering = descending count (PhlF/P2 most universal first).
TOP5_COLORS = {
    "PhlF/P2": "#0C7AB0",   # ACCENT_BLUE_DARK
    "PsrA/R1": "#A50026",   # paper toxic-extreme red
    "AmtR/A1": "#6A4C9C",   # purple
    "BetI/E1": "#1A9850",   # PASS_GREEN
    "QacR/Q2": "#C99800",   # ACCENT_YELLOW_DARK
}

PAPER_YELLOW_DOTS = [
    {"target": "0x2B", "size": 5, "circuit": 36.5869,  "growth": 0.7609},
    {"target": "0x17", "size": 6, "circuit": 125.7974, "growth": 0.7590},
    {"target": "0x6D", "size": 7, "circuit": 5.2859,   "growth": 0.6883},
]

CANVAS_W_MM = 280
CANVAS_H_MM = 200


def _parse_source_list(src) -> list[str]:
    s = str(src).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            return [s]
    return [s]


def _best_per_target() -> pd.DataFrame:
    fronts = L.load_pareto_fronts("G3").copy()
    fronts["circuit_score"] = np.exp(fronts["circuit_log"])
    mask = ((fronts["circuit_score"] >= CIRC_SCORE_FLOOR)
            & (fronts["toxicity"] >= TOX_FLOOR))
    buildable = fronts[mask].copy()
    buildable["sources_list"] = buildable["source"].apply(_parse_source_list)
    rows: list[dict] = []
    for _, r in buildable.iterrows():
        for tgt in r["sources_list"]:
            rows.append({**r.to_dict(), "target": tgt})
    long = pd.DataFrame(rows)
    best = (long.sort_values("circuit_log", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = (best.sort_values("circuit_log", ascending=False)
                .reset_index(drop=True))
    best["rank"]         = best.index + 1
    best["npn_class"]    = best["target"].apply(S.npn_class_of)
    best["target_label"] = best["target"]
    return best


def _render_parts_cell(ax, x0: float, y: float, parts: list[str], *,
                       mode: str) -> None:
    """Render the part list at (x0, y) with top-5 highlighted.

    mode: "color" — top-5 parts get coloured TEXT, others grey.
          "box"   — top-5 parts get coloured BOX OUTLINE, text stays grey.
    """
    # Monospace character width calibration for fontsize=5.0 inside this
    # ax2's data coordinates (xlim 0..16, panel ~140 mm wide). Empirical
    # value tuned so the longest 7-part 7-reg row fits without clipping.
    CHAR_W   = 0.115
    SEP      = "  ·  "
    SEP_W    = len(SEP) * CHAR_W
    BOX_PAD_X = 0.04
    BOX_PAD_Y = 0.30
    BOX_LW    = 0.7

    x = x0
    for j, part in enumerate(parts):
        is_top5  = part in TOP5_COLORS
        color    = TOP5_COLORS[part] if (is_top5 and mode == "color") else TEXT_GREY
        weight   = "bold" if is_top5 else "normal"
        ax.text(x, y, part,
                 fontsize=5.0, va="center", ha="left",
                 family="monospace", color=color, fontweight=weight,
                 zorder=4)
        part_w = len(part) * CHAR_W
        if mode == "box" and is_top5:
            rect = Rectangle(
                (x - BOX_PAD_X, y - BOX_PAD_Y),
                part_w + 2 * BOX_PAD_X, 2 * BOX_PAD_Y,
                facecolor="none", edgecolor=TOP5_COLORS[part],
                linewidth=BOX_LW, zorder=3,
            )
            ax.add_patch(rect)
        x += part_w
        if j < len(parts) - 1:
            ax.text(x, y, SEP,
                     fontsize=5.0, va="center", ha="left",
                     family="monospace", color=TEXT_GREY, zorder=4)
            x += SEP_W


def _render_top5_legend(ax, x0: float, y: float) -> None:
    """Compact legend strip showing the 5 colour↔part mappings.

    Placed inside ax2 at the top of the right pane."""
    CHAR_W = 0.115
    PAD    = 0.20    # gap between legend entries
    SWATCH_W = 0.30  # tiny coloured rect / line indicator
    BOX_PAD_X = 0.04
    BOX_PAD_Y = 0.22

    ax.text(x0, y, "Top-5 parts:",
             fontsize=5.5, va="center", ha="left",
             color=TEXT_GREY, fontweight="bold")
    x = x0 + len("Top-5 parts:") * CHAR_W + PAD * 1.5

    for part, color in TOP5_COLORS.items():
        # Small swatch (box outline) on left of label so legend
        # visually reads as "this is the marker".
        rect = Rectangle(
            (x, y - BOX_PAD_Y),
            SWATCH_W, 2 * BOX_PAD_Y,
            facecolor="none", edgecolor=color, linewidth=0.9,
        )
        ax.add_patch(rect)
        ax.text(x + SWATCH_W + 0.06, y, part,
                 fontsize=5.5, va="center", ha="left",
                 family="monospace", color=color, fontweight="bold")
        x += SWATCH_W + 0.06 + len(part) * CHAR_W + PAD


def build_panel(*, mode: str, out_name: str) -> None:
    """mode ∈ {color, box}; out_name e.g. 'panel_c_v2'."""
    use_style()

    knees   = L.load_pareto_knees("G3")
    knees["npn_class"] = knees["source"].apply(S.npn_class_of)

    best_pt = _best_per_target()
    n_targets = len(best_pt)

    for yd in PAPER_YELLOW_DOTS:
        yd["circuit_log"] = float(np.log(yd["circuit"]))
        yd["passes_circ"] = yd["circuit_log"] >= CIRC_LOG_FLOOR
        yd["passes_tox"]  = yd["growth"]      >= TOX_FLOOR

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM),
        gridspec_kw={"width_ratios": [1.50, 1.55]},
    )
    fig.subplots_adjust(
        left=0.05, right=0.985, top=0.92, bottom=0.08, wspace=0.10,
    )

    # =================================================================
    # LEFT pane (unchanged from canonical Panel C)
    # =================================================================
    for npn in S.NPN_CLASS_ORDER:
        sub = knees[knees["npn_class"] == npn]
        if len(sub) == 0:
            continue
        ax1.scatter(
            sub["knee_A_circuit_log"], sub["knee_A_toxicity"],
            s=14, c=S.NPN_CLASS_COLORS[npn],
            edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3,
            label=f"NPN {npn} (n={len(sub)})",
        )

    yd_min_circ = min(y["circuit_log"] for y in PAPER_YELLOW_DOTS)
    yd_min_tox  = min(y["growth"]      for y in PAPER_YELLOW_DOTS)
    xlim_min = min(knees["knee_A_circuit_log"].min() - 0.3,
                   yd_min_circ - 0.5, -0.5)
    xlim_max = max(knees["knee_A_circuit_log"].max() + 0.4,
                   best_pt["circuit_log"].max() + 0.4, 9.5)
    ylim_min = min(0.45, yd_min_tox - 0.03,
                   best_pt["toxicity"].min() - 0.03)
    ylim_max = max(knees["knee_A_toxicity"].max() + 0.02, 1.0)
    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(ylim_min, ylim_max)

    rect = Rectangle(
        (CIRC_LOG_FLOOR, TOX_FLOOR),
        xlim_max - CIRC_LOG_FLOOR, ylim_max - TOX_FLOOR,
        facecolor=PASTEL["green"], alpha=0.22,
        edgecolor=DARK_GRAY, linewidth=0.5, linestyle="--", zorder=1,
    )
    ax1.add_patch(rect)
    ax1.axvline(CIRC_LOG_FLOOR, color=DARK_GRAY, linestyle="--",
                 linewidth=0.5, alpha=0.6, zorder=2)
    ax1.axhline(TOX_FLOOR,  color=DARK_GRAY, linestyle="--",
                 linewidth=0.5, alpha=0.6, zorder=2)
    ax1.text(CIRC_LOG_FLOOR + 0.10, ylim_max - 0.005,
              "buildable (paper MLP gate)",
              fontsize=6.0, color=TEXT_GREY, va="top", ha="left",
              fontweight="normal")

    for _, r in best_pt.iterrows():
        ax1.scatter(
            r["circuit_log"], r["toxicity"],
            s=48, facecolor="none", edgecolor=DARK_GRAY,
            linewidth=0.9, zorder=5,
        )

    for yd in PAPER_YELLOW_DOTS:
        ax1.scatter(
            yd["circuit_log"], yd["growth"],
            marker="*", s=220,
            c=ACCENT_YELLOW, edgecolor=ACCENT_YELLOW_DARK,
            linewidth=0.9, zorder=7,
        )
    label_offsets = {
        "0x2B": (-12, -28),
        "0x17": (12, 16),
        "0x6D": (-4, 30),
    }
    for yd in PAPER_YELLOW_DOTS:
        dx, dy = label_offsets.get(yd["target"], (8, 8))
        ha = "right" if dx < 0 else "left"
        ax1.annotate(
            f"{yd['target']} ({yd['size']}-reg)\n"
            f"c={yd['circuit']:.1f}, g={yd['growth']:.2f}",
            (yd["circuit_log"], yd["growth"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=5.5, color=DARK_GRAY,
            fontweight="normal", zorder=8, ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor=ACCENT_YELLOW_DARK,
                       linewidth=0.6, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=DARK_GRAY, linewidth=0.35),
        )

    ax1.set_xlabel(
        "circuit_log  (higher = better logic margin)\n"
        "small dots: 215 knee designs  ·  rings: per-target best  "
        "·  gold stars: paper Fig. 3 simulator yellow-dots",
        fontsize=6.0, color=TEXT_GREY, labelpad=2.5,
    )
    ax1.set_ylabel("growth score  (higher = healthier cell)",
                    fontsize=6.0, color=TEXT_GREY, labelpad=2.5)
    ax1.set_title(
        f"{len(knees)} knee designs by NPN class\n"
        f"{n_targets} per-target best designs ringed  ·  "
        f"buildable: circuit_score >= 2  AND  growth >= 0.5",
        loc="left", pad=6,
        fontsize=6.5, fontweight="normal", color=TEXT_GREY,
    )
    leg = ax1.legend(
        loc="lower right", frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fontsize=5.0,
        handletextpad=0.4, ncol=2, columnspacing=0.6,
        borderpad=0.4, labelspacing=0.25,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT_GREY)
    ax1.grid(True, linewidth=0.25, alpha=0.35,
              color=DARK_GRAY, zorder=0)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax1.spines[spine].set_color(DARK_GRAY)
        ax1.spines[spine].set_linewidth(0.6)
    ax1.tick_params(
        axis="both", which="major",
        color=DARK_GRAY, width=0.6, length=2.5,
        labelsize=5.5, labelcolor=DARK_GRAY,
    )

    # =================================================================
    # RIGHT pane (modified — top-5 highlight in parts column)
    # =================================================================
    table_xlim = 16.0
    ax2.set_xlim(0, table_xlim)
    ax2.set_ylim(-3.6, n_targets + 6.0)
    ax2.invert_yaxis()
    ax2.axis("off")

    # Top-5 legend strip — sits ABOVE the header rule.
    _render_top5_legend(ax2, x0=0.1, y=-2.4)

    cols_x  = [0.1, 1.1, 2.2, 3.2, 4.6, 5.9, 7.1]
    headers = ["rank", "target", "size", "circuit_log",
               "circuit_score", "growth", "part_names"]
    header_y = -0.4
    for x, h in zip(cols_x, headers):
        ax2.text(x, header_y, h,
                  fontsize=5.5, fontweight="normal",
                  color=TEXT_GREY, va="bottom", ha="left")
    ax2.axhline(header_y + 0.2, color=DARK_GRAY, linewidth=0.5)

    for i, (_, r) in enumerate(best_pt.iterrows()):
        y = i + 0.4
        npn_color = S.NPN_CLASS_COLORS.get(r["npn_class"], "#999999")
        ax2.text(cols_x[0], y, f"#{int(r['rank'])}",
                  fontsize=5.5, va="center",
                  fontweight="normal", color=TEXT_GREY)
        ax2.add_patch(Rectangle(
            (cols_x[1] - 0.08, y - 0.18), 0.16, 0.36,
            facecolor=npn_color, edgecolor="none",
        ))
        ax2.text(cols_x[1] + 0.15, y, r["target_label"],
                  fontsize=5.5, va="center", color=TEXT_GREY)
        ax2.text(cols_x[2], y, f"{int(r['gate_count'])}-reg",
                  fontsize=5.5, va="center", color=TEXT_GREY)
        ax2.text(cols_x[3], y, f"{r['circuit_log']:.2f}",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        ax2.text(cols_x[4], y, f"{r['circuit_score']:.1f}",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        ax2.text(cols_x[5], y, f"{r['toxicity']:.3f}",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        parts = (ast.literal_eval(r["part_names"])
                 if isinstance(r["part_names"], str)
                 else list(r["part_names"]))
        _render_parts_cell(ax2, cols_x[6], y, parts, mode=mode)

    # --- Paper Fig. 3 yellow-dot reference strip ---
    strip_y0 = n_targets + 1.6
    ax2.axhline(strip_y0 - 0.6, color=MID_GRAY, linewidth=0.4,
                 linestyle="--")
    ax2.text(cols_x[0], strip_y0 - 0.15,
              "Paper Fig. 3 yellow-dots (sim labels):",
              fontsize=5.8, fontweight="normal", va="bottom",
              ha="left", color=ACCENT_YELLOW_DARK)

    x_circ_chk = cols_x[6] + 0.6
    x_circ_val = cols_x[6] + 2.4
    x_tox_chk  = cols_x[6] + 4.4
    x_tox_val  = cols_x[6] + 6.2

    yd_header_y = strip_y0 + 0.5
    yd_main_x = [cols_x[1], cols_x[2], cols_x[3], cols_x[4], cols_x[5]]
    yd_headers_main = ["target", "size", "circuit_log",
                        "circuit_score", "growth"]
    for x, h in zip(yd_main_x, yd_headers_main):
        ax2.text(x, yd_header_y, h,
                  fontsize=5.0, fontweight="normal",
                  va="bottom", ha="left", color=TEXT_GREY)
    ax2.text(cols_x[6], yd_header_y, "vs buildable threshold:",
              fontsize=5.0, fontweight="normal",
              va="bottom", ha="left", color=TEXT_GREY)
    ax2.axhline(yd_header_y + 0.18, color=MID_GRAY, linewidth=0.4)

    for i, yd in enumerate(PAPER_YELLOW_DOTS):
        y = strip_y0 + 1.15 + i * 0.7
        ax2.scatter(
            [cols_x[1] - 0.20], [y],
            marker="*", s=60,
            c=ACCENT_YELLOW, edgecolor=ACCENT_YELLOW_DARK,
            linewidth=0.6, zorder=4, clip_on=False,
        )
        ax2.text(cols_x[1] + 0.15, y, yd["target"],
                  fontsize=5.5, va="center", color=TEXT_GREY)
        ax2.text(cols_x[2], y, f"{yd['size']}-reg",
                  fontsize=5.5, va="center", color=TEXT_GREY)
        ax2.text(cols_x[3], y, f"{yd['circuit_log']:.2f}",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        ax2.text(cols_x[4], y, f"{yd['circuit']:.1f}",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        ax2.text(cols_x[5], y, f"{yd['growth']:.3f}",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        circ_ok = "PASS" if yd["passes_circ"] else "FAIL"
        tox_ok  = "PASS" if yd["passes_tox"]  else "FAIL"
        circ_color = PASS_GREEN if yd["passes_circ"] else FAIL_RED
        tox_color  = PASS_GREEN if yd["passes_tox"]  else FAIL_RED
        ax2.text(x_circ_chk, y, "circ:",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        ax2.text(x_circ_val, y, circ_ok,
                  fontsize=5.5, va="center", family="monospace",
                  color=circ_color, fontweight="normal")
        ax2.text(x_tox_chk, y, "tox:",
                  fontsize=5.5, va="center", family="monospace",
                  color=TEXT_GREY)
        ax2.text(x_tox_val, y, tox_ok,
                  fontsize=5.5, va="center", family="monospace",
                  color=tox_color, fontweight="normal")

    mode_label = {"color": "v2 — coloured TEXT",
                   "box":   "v3 — coloured BOX OUTLINE"}[mode]
    ax2.set_title(
        f"Best buildable design per target function (n={n_targets})\n"
        f"circuit_score >= {CIRC_SCORE_FLOOR:.0f}  AND  "
        f"growth >= {TOX_FLOOR:.1f}  (paper MLP-gate thresholds)  ·  "
        f"{mode_label}",
        loc="left", pad=6,
        fontsize=6.5, fontweight="normal", color=TEXT_GREY,
    )

    out_base = PANELS_VEC / out_name
    written = save_panel(fig, out_base, dpi=600, close=True)
    print(f"{out_name} written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel(mode="color", out_name="panel_c_v2")
    build_panel(mode="box",   out_name="panel_c_v3")
