"""Figure S04 — Panel C (DRAFT): knee-design scatter + per-target best-design table.

[Superseded 2026-05-27 — the canonical Panel C is now the knee-vs-maxcirc
visualisation rendered by build_panel_c.py. This script is preserved as
panel_c_draft.{pdf,svg,png} for provenance + comparison.]


Update (2026-05-26): switched from analysis-side Path-A thresholds
(circuit_log >= median + 0.5 sigma, growth >= 0.85) to the paper's
**MLP-gate thresholds** (circuit_score >= 2, growth >= 0.5). These are
the same gates the upstream NNGGA pipeline uses to decide which MLP
picks go to the simulator (NNGGA designs parallel GPUs.py L1546, L1553).

The right-side table now shows the **single highest-scoring buildable
design for each of the 20 target functions** (one row per target),
sorted by circuit_log descending — replacing the previous 24-row
top-portfolio-across-targets table, which crowded around a few high-
performing topologies. The per-target framing surfaces more biology-
relevant diversity in which topology / size / part-assignment wins for
each Boolean function.

Two sub-panes:

  LEFT  — scatter of all 215 G3 knee designs (knee_circuit_log vs
          knee_toxicity), coloured by NPN equivalence class. Buildable
          region shaded pastel green (new paper-MLP threshold floors).
          The 20 per-target best designs are overlaid as dark-rimmed
          open circles at their actual (circuit_log, toxicity) — these
          are NOT the knees, they are the max-circuit picks per target
          subject to the MLP feasibility gate. Paper Fig. 3 yellow-dots
          (0x2B/0x17/0x6D) overlaid as ACCENT_YELLOW stars with
          callouts.

  RIGHT — best-design-per-target table: rank · target · size ·
          circuit_log · circuit_score · growth · part_names. 20 rows;
          NPN-coloured rectangle precedes each target. Below the table:
          paper Fig. 3 yellow-dot reference strip with PASS / FAIL
          badges against the new MLP-gate thresholds.

Outputs:
  panels/vector/panel_c.pdf | .svg
  panels/raster/panel_c.png

Run:
    python \
        figures/setFinal/figS13/scripts/build_panel_c.py
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict, deque
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

sys.path.insert(0, str(REPO_ROOT / "topology" / "figures"))
import _loaders as L  # noqa: E402
import _style as S    # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    ACCENT_YELLOW, ACCENT_YELLOW_DARK,
    DARK_GRAY, MID_GRAY, LIGHT_GRAY, PASTEL,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS13"
              / "panels" / "vector")

TEXT_GREY = "#666666"
PASS_GREEN = "#1A9850"
FAIL_RED   = "#D73027"

# Paper MLP-gate thresholds (from upstream NNGGA pipeline).
CIRC_SCORE_FLOOR = 2.0
CIRC_LOG_FLOOR   = float(np.log(CIRC_SCORE_FLOOR))   # ~0.6931
TOX_FLOOR        = 0.5

# Top-5 parts across the n=20 per-target best designs (PhlF/P2 20/20,
# PsrA/R1 19/20, AmtR/A1 16/20, BetI/E1 14/20, QacR/Q2 13/20). Each gets
# a distinct paper-consistent hue so a reader can scan rows for "where
# does part X appear?". Order = descending presence count.
TOP5_COLORS = {
    "PhlF/P2": "#0C7AB0",   # ACCENT_BLUE_DARK
    "PsrA/R1": "#A50026",   # paper toxic-extreme red
    "AmtR/A1": "#6A4C9C",   # purple
    "BetI/E1": "#1A9850",   # PASS_GREEN
    "QacR/Q2": "#C99800",   # ACCENT_YELLOW_DARK
}

# 20-part TetR-family library; index ↔ name mapping (paper Methods).
# Recovered by cross-referencing gate_assignments with part_names in
# G3 Pareto-front data; verified against the 0x2B yellow-dot pickle.
LIBRARY_PART_NAMES = {
    0:  "AmeR/F1",   1:  "AmtR/A1",
    2:  "BetI/E1",
    3:  "BM3R1/B1",  4:  "BM3R1/B2",  5:  "BM3R1/B3",
    6:  "HlyIIR/H1",
    7:  "IcaRA/I1",
    8:  "LitR/L1",
    9:  "LmrA/N1",
    10: "PhlF/P1",   11: "PhlF/P2",   12: "PhlF/P3",
    13: "PsrA/R1",
    14: "QacR/Q1",   15: "QacR/Q2",
    16: "SrpR/S1",   17: "SrpR/S2",   18: "SrpR/S3",   19: "SrpR/S4",
}

# Paper Fig. 3 yellow-dot designs (sim-rank-1 of each exemplar).
# slot_d2o = depth-to-output per regulator slot, computed once from the
# optimal_topology pickles in $DEEPCIRC_EXEMPLARS/0x{2B,17,6D}/. Embedded
# here so the paper_figures pipeline stays self-contained.
PAPER_YELLOW_DOTS = [
    {"target": "0x2B", "size": 5, "circuit": 36.5869,  "growth": 0.7609,
     "perm": [2, 19, 15, 13, 12],
     "slot_d2o": [2, 1, 2, 2, 1]},
    {"target": "0x17", "size": 6, "circuit": 125.7974, "growth": 0.7590,
     "perm": [15, 11, 2, 13, 19, 1],
     "slot_d2o": [1, 2, 3, 2, 4, 4]},
    {"target": "0x6D", "size": 7, "circuit": 5.2859,   "growth": 0.6883,
     "perm": [1, 19, 13, 11, 0, 2, 3],
     "slot_d2o": [3, 2, 2, 1, 3, 2, 1]},
]


CANVAS_W_MM = 280
CANVAS_H_MM = 200


def _parse_source_list(src) -> list[str]:
    """Source column may be a JSON-like list ('["0x2B","0x4D"]') or a
    single token ('0x2B'). Always return list[str]."""
    s = str(src).strip()
    if s.startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            return [s]
    return [s]


def _compute_slot_d2o(topo: dict) -> list[int]:
    """Return d2o (depth-to-output) per regulator slot, in slot-index
    order (ascending non-IO node id, matching gate_assignments)."""
    nodes = {n["id"]: n["type"] for n in topo["nodes"]}
    rev: dict = defaultdict(list)
    for u, v in topo["edges"]:
        rev[v].append(u)
    out_nodes = [nid for nid, t in nodes.items() if t == "OUT"]
    d2o: dict[int, int] = {}
    q = deque([(o, 0) for o in out_nodes])
    seen = set(out_nodes)
    while q:
        n, d = q.popleft()
        if n not in d2o or d < d2o[n]:
            d2o[n] = d
        for p in rev[n]:
            if p not in seen:
                seen.add(p); q.append((p, d + 1)); d2o[p] = d + 1
    regs = sorted([nid for nid, t in nodes.items() if t not in ("IN", "OUT")])
    return [d2o[r] for r in regs]


def _reorder_parts_by_position(parts: list[str], slot_d2o: list[int]
                                 ) -> list[str]:
    """Reorder a part list so input-proximal slots come first (high d2o
    → low d2o). Stable secondary sort by original slot index keeps
    parts at the same d2o in their canonical node-id order."""
    indexed = list(enumerate(parts))
    indexed.sort(key=lambda ip: (-slot_d2o[ip[0]], ip[0]))
    return [p for _, p in indexed]


def _render_parts_cell(ax, x0: float, y: float, parts: list[str]) -> None:
    """Render the part list at (x0, y) with top-5 parts in coloured bold
    text (non-top-5 stay grey). Monospace character width is calibrated
    empirically for ax2's data coordinates (xlim 0..16, fontsize 5.0)."""
    CHAR_W = 0.115
    SEP    = "  ·  "
    SEP_W  = len(SEP) * CHAR_W
    x = x0
    for j, part in enumerate(parts):
        is_top5 = part in TOP5_COLORS
        color   = TOP5_COLORS[part] if is_top5 else TEXT_GREY
        weight  = "bold" if is_top5 else "normal"
        ax.text(x, y, part,
                 fontsize=5.0, va="center", ha="left",
                 family="monospace", color=color, fontweight=weight)
        x += len(part) * CHAR_W
        if j < len(parts) - 1:
            ax.text(x, y, SEP,
                     fontsize=5.0, va="center", ha="left",
                     family="monospace", color=TEXT_GREY)
            x += SEP_W


def _render_top5_legend(ax, x0: float, y: float) -> None:
    """Compact legend strip above the table header. Part names are
    shown directly in their highlight colour — no swatch boxes."""
    CHAR_W = 0.115
    PAD    = 0.30
    ax.text(x0, y, "Top-5 parts:",
             fontsize=5.5, va="center", ha="left",
             color=TEXT_GREY, fontweight="bold")
    x = x0 + len("Top-5 parts:") * CHAR_W + PAD
    for part, color in TOP5_COLORS.items():
        ax.text(x, y, part,
                 fontsize=5.5, va="center", ha="left",
                 family="monospace", color=color, fontweight="bold")
        x += len(part) * CHAR_W + PAD


def _build_best_per_target(fronts: pd.DataFrame) -> pd.DataFrame:
    """Filter the full Pareto-front set by paper MLP thresholds, expand
    multi-source rows so each (design, target) pair contributes a row,
    then keep the single highest-circuit_log design per target."""
    fronts = fronts.copy()
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

    # Highest circuit_log per target subject to growth >= floor.
    best = (long.sort_values("circuit_log", ascending=False)
                .groupby("target", as_index=False)
                .first())
    best = (best.sort_values("circuit_log", ascending=False)
                .reset_index(drop=True))
    best["rank"]         = best.index + 1
    best["npn_class"]    = best["target"].apply(S.npn_class_of)
    best["target_label"] = best["target"]
    return best


def build_panel() -> None:
    use_style()

    graphs = json.loads((REPO_ROOT / "data" / "topology_g3"
                         / "topology_graphs.json").read_text())
    topo_by_id = {t["topology_id"]: t for t in graphs["topologies"]}

    knees   = L.load_pareto_knees("G3")
    fronts  = L.load_pareto_fronts("G3")
    knees["npn_class"] = knees["source"].apply(S.npn_class_of)

    best_pt = _build_best_per_target(fronts)
    n_targets = len(best_pt)

    # Pre-compute pass/fail flags for paper yellow-dots against the new
    # thresholds.
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

    # -----------------------------------------------------------------
    # LEFT pane (ax1): knee scatter + buildable region + best-per-target
    # rings + paper yellow-dots
    # -----------------------------------------------------------------
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

    # Extend axes so growth = 0.50 floor is visible (was floored at
    # 0.65 under the old buildable threshold of 0.85).
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

    # Buildable region — pastel green shading at NEW paper-MLP floors.
    # All region edges + threshold floor lines render in LIGHT_GRAY so
    # they read as a uniform background grid, not load-bearing dark
    # strokes. (Previously mixed DARK_GRAY/MID_GRAY produced an
    # off-coloured grid feel.)
    rect = Rectangle(
        (CIRC_LOG_FLOOR, TOX_FLOOR),
        xlim_max - CIRC_LOG_FLOOR, ylim_max - TOX_FLOOR,
        facecolor=PASTEL["green"], alpha=0.22,
        edgecolor=LIGHT_GRAY, linewidth=0.5, linestyle="--", zorder=1,
    )
    ax1.add_patch(rect)
    ax1.axvline(CIRC_LOG_FLOOR, color=LIGHT_GRAY, linestyle="--",
                 linewidth=0.5, alpha=1.0, zorder=2)
    ax1.axhline(TOX_FLOOR,  color=LIGHT_GRAY, linestyle="--",
                 linewidth=0.5, alpha=1.0, zorder=2)
    ax1.text(CIRC_LOG_FLOOR + 0.10, ylim_max - 0.005,
              "buildable (paper MLP gate)",
              fontsize=6.0, color=TEXT_GREY, va="top", ha="left",
              fontweight="normal")

    # Paper Fig. 3 yellow-dots — gold stars with callouts.
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
            yd["target"],
            (yd["circuit_log"], yd["growth"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=6.0, color=DARK_GRAY,
            fontweight="bold", zorder=8, ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor=ACCENT_YELLOW_DARK,
                       linewidth=0.6, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=DARK_GRAY, linewidth=0.35),
        )

    ax1.set_xlabel(
        "circuit_log  (higher = better logic margin)\n"
        "small dots: 215 knee designs  ·  "
        "gold stars: paper Fig. 3 simulator yellow-dots",
        fontsize=6.0, color=TEXT_GREY, labelpad=2.5,
    )
    ax1.set_ylabel("growth score  (higher = healthier cell)",
                    fontsize=6.0, color=TEXT_GREY, labelpad=2.5)
    ax1.set_title(
        f"{len(knees)} knee designs by NPN class  ·  "
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
    ax1.grid(True, linewidth=0.25, alpha=1.0,
              color=LIGHT_GRAY, zorder=0)
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

    # -----------------------------------------------------------------
    # RIGHT pane (ax2): best-per-target table + paper-anchor strip
    # -----------------------------------------------------------------
    table_xlim = 16.0
    ax2.set_xlim(0, table_xlim)
    ax2.set_ylim(-3.6, n_targets + 6.0)
    ax2.invert_yaxis()
    ax2.axis("off")

    # Top-5 colour legend above the table header.
    _render_top5_legend(ax2, x0=0.1, y=-2.4)

    # 7 columns: rank, target, size, circuit_log, circuit_score, growth, parts
    cols_x  = [0.1, 1.1, 2.2, 3.2, 4.6, 5.9, 7.1]
    headers = ["rank", "target", "size", "c_log", "c_score", "growth",
               "part_names (input → output, position-informed)"]
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
        slot_d2o = _compute_slot_d2o(topo_by_id[r["topology_id"]])
        parts = _reorder_parts_by_position(parts, slot_d2o)
        _render_parts_cell(ax2, cols_x[6], y, parts)

    # --- Reference strip: paper Fig. 3 yellow-dots ---
    strip_y0 = n_targets + 1.6
    ax2.axhline(strip_y0 - 0.6, color=MID_GRAY, linewidth=0.4,
                 linestyle="--")
    ax2.text(cols_x[0], strip_y0 - 0.15,
              "Paper Fig. 3 yellow-dots  "
              "(scores below are simulator ground truth, NOT MLP predictions):",
              fontsize=5.8, fontweight="normal", va="bottom",
              ha="left", color=ACCENT_YELLOW_DARK)

    yd_header_y = strip_y0 + 0.5
    yd_main_x = [cols_x[1], cols_x[2], cols_x[3], cols_x[4], cols_x[5]]
    yd_headers_main = ["target", "size", "c_log", "c_score", "growth"]
    for x, h in zip(yd_main_x, yd_headers_main):
        ax2.text(x, yd_header_y, h,
                  fontsize=5.0, fontweight="normal",
                  va="bottom", ha="left", color=TEXT_GREY)
    ax2.text(cols_x[6], yd_header_y,
              "part_names (input → output, position-informed)",
              fontsize=5.0, fontweight="normal",
              va="bottom", ha="left", color=TEXT_GREY)
    ax2.axhline(yd_header_y + 0.18, color=MID_GRAY, linewidth=0.4)

    for i, yd in enumerate(PAPER_YELLOW_DOTS):
        y = strip_y0 + 1.15 + i * 0.7
        # Centre the star under the rank column ("#1..#20" above).
        ax2.scatter(
            [cols_x[0] + 0.10], [y],
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
        # Position-informed parts list (input → output) for the
        # yellow-dot's regulator slots.
        yd_parts = [LIBRARY_PART_NAMES[i] for i in yd["perm"]]
        yd_parts = _reorder_parts_by_position(yd_parts, yd["slot_d2o"])
        _render_parts_cell(ax2, cols_x[6], y, yd_parts)

    ax2.set_title(
        f"Best buildable design per target function (n={n_targets})\n"
        f"circuit_score >= {CIRC_SCORE_FLOOR:.0f}  AND  "
        f"growth >= {TOX_FLOOR:.1f}  (paper MLP-gate thresholds)",
        loc="left", pad=6,
        fontsize=6.5, fontweight="normal", color=TEXT_GREY,
    )

    out_base = PANELS_VEC / "panel_c_draft"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel C (draft) written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
