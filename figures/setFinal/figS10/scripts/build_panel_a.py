"""Figure S10 — Panel A: interpretability pipeline overview.

Wide horizontal pipeline strip showing how this paper-supplementary
figure family extracts design rules from a trained DeepCirc model.

  ┌────────────────┐         ┌────┐ ┌────┐ ┌────┐ ┌────┐
  │                │   ──┐   │ 1  │ │ 2  │ │ 3  │ │ 4  │
  │   TRAINED      │     │   │... │ │... │ │... │ │... │
  │   DEEPCIRC     │  ───┼─→ │    │ │    │ │    │ │    │
  │                │     │   │S10 │ │S10d│ │S12 │ │S14 │
  │   20 targets × │   ──┘   │b,c │ │S11 │ │S13 │ │S15 │
  │   259 topologies         └────┘ └────┘ └────┘ └────┘
  └────────────────┘
     left box                fanned    4 numbered lens thumbnails
                             arrows    each with an abstract glyph

The four lenses are:
  1. Population cartography    (figS10 b, c)
  2. Structural motifs         (figS10 d, figS11)
  3. Cohort selection          (figS12, figS13)
  4. Composition → function    (figS14, figS15)

Outputs (vector + raster):
  panels/vector/panel_a.pdf | .svg
  panels/raster/panel_a.png

Run:
    python \\
        figures/setFinal/figS10/scripts/build_panel_a.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Rectangle, FancyArrowPatch, FancyBboxPatch, Circle,
)

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    ACCENT_BLUE, ACCENT_BLUE_DARK, DARK_GRAY, MID_GRAY, LIGHT_GRAY,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS10"
              / "panels" / "vector")

# Match the manifest slot 1:1 so the panel renders at native size.
CANVAS_W_MM = 232
CANVAS_H_MM = 50

BOX_BORDER = "#bbbbbb"


# --- glyph helpers ---------------------------------------------------------

def _glyph_population(ax, cx, cy, w, h):
    """Right-tail-skewed distribution silhouette."""
    n = 9
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    heights = np.array([0.1, 0.18, 0.32, 0.50, 0.68, 0.85, 0.95, 0.55, 0.22])
    bar_w = w / (n + 1.0) * 0.9
    for x, hh in zip(xs, heights):
        bar_h = hh * h
        ax.add_patch(Rectangle(
            (x - bar_w / 2, cy + h / 2 - bar_h), bar_w, bar_h,
            facecolor=ACCENT_BLUE, edgecolor="none",
        ))
    ax.plot([cx - w / 2 - 0.5, cx + w / 2 + 0.5],
             [cy + h / 2, cy + h / 2],
             color=DARK_GRAY, linewidth=0.6, solid_capstyle="round")


def _glyph_motif(ax, cx, cy, w, h):
    """Three non-linear structural motifs in a triangle layout (2 on
    top, 1 centred below). Sized to USE the glyph's full extent so
    they don't look cramped in the thumbnail.

    Motifs:
      top-left  : feed-forward loop  A→B, B→C, A→C   (3-node, 3 edges)
      top-right : 3-cycle            A→B, B→C, C→A   (3-node, closed)
      bottom    : diamond / bi-fan   A→B, A→C, B→D, C→D   (4-node)
    """
    r = 0.8
    DX = 3.0    # horizontal half-extent of each motif
    DY = 2.0    # vertical half-extent of each motif

    def _edge(x0, y0, x1, y1, *, rad=0.0):
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        cs = "arc3" if abs(rad) < 1e-6 else f"arc3,rad={rad}"
        ax.add_patch(FancyArrowPatch(
            (x0 + ux * r, y0 + uy * r),
            (x1 - ux * r, y1 - uy * r),
            arrowstyle="-|>", mutation_scale=5,
            color=DARK_GRAY, linewidth=0.7,
            shrinkA=0, shrinkB=0, zorder=2,
            connectionstyle=cs,
        ))

    def _node(x, y):
        ax.add_patch(Circle((x, y), r,
                              facecolor=ACCENT_BLUE,
                              edgecolor=ACCENT_BLUE_DARK,
                              linewidth=0.45, zorder=3))

    # Triangle layout anchors — use larger horizontal & vertical spread
    # so each motif occupies a proper portion of the glyph area.
    top_l = (cx - 8.0, cy - 3.5)
    top_r = (cx + 8.0, cy - 3.5)
    bot_c = (cx,       cy + 3.6)

    # --- Top-left: fan-out  A→B, A→C  (sideways Y) --------------------
    # Visually distinct from the right-hand triangular 3-cycle: one
    # source branching to two outputs.
    tlx, tly = top_l
    m1a = (tlx - DX, tly)                  # source (left)
    m1b = (tlx + DX, tly - DY * 0.8)       # output top
    m1c = (tlx + DX, tly + DY * 0.8)       # output bot
    _edge(*m1a, *m1b)
    _edge(*m1a, *m1c)
    _node(*m1a); _node(*m1b); _node(*m1c)

    # --- Top-right: 3-cycle  A→B, B→C, C→A (closed loop) --------------
    trx, try_y = top_r
    m2a = (trx,      try_y - DY)            # top
    m2b = (trx + DX, try_y + DY * 0.6)      # bot-right
    m2c = (trx - DX, try_y + DY * 0.6)      # bot-left
    _edge(*m2a, *m2b)
    _edge(*m2b, *m2c)
    _edge(*m2c, *m2a)
    _node(*m2a); _node(*m2b); _node(*m2c)

    # --- Bottom: diamond / bi-fan A→B, A→C, B→D, C→D ------------------
    bx, by = bot_c
    m3a = (bx,        by - DY * 1.0)   # top   (input)
    m3b = (bx - DX,   by)              # mid-left
    m3c = (bx + DX,   by)              # mid-right
    m3d = (bx,        by + DY * 1.0)   # bot   (output)
    _edge(*m3a, *m3b)
    _edge(*m3a, *m3c)
    _edge(*m3b, *m3d)
    _edge(*m3c, *m3d)
    _node(*m3a); _node(*m3b); _node(*m3c); _node(*m3d)


def _glyph_cohort(ax, cx, cy, w, h):
    """Dense cluster scatter — mirrors figS13's pareto-optimal cloud
    shape (elliptical Gaussian, ~80 points) but in grey, with a small
    number of points randomly recoloured BLUE and BLACK as cohort
    exemplars. All dots share the same marker size."""
    rng = np.random.default_rng(2026)
    n = 200
    # Bivariate Gaussian, slightly elongated horizontally — emulates
    # figS13's cluster shape (more spread along circuit_score).
    sigma_x = w * 0.22
    sigma_y = h * 0.18
    xs = cx + rng.normal(0.0, sigma_x, n)
    ys = cy + rng.normal(0.0, sigma_y, n)
    # Clip to glyph box so we don't bleed past the thumbnail edges.
    xs = np.clip(xs, cx - w / 2 + 0.4, cx + w / 2 - 0.4)
    ys = np.clip(ys, cy - h / 2 + 0.4, cy + h / 2 - 0.4)

    DOT_S = 3.5
    # Pick a handful of random indices to recolour as exemplars.
    n_black = 3
    n_blue  = 4
    picks = rng.choice(n, size=n_black + n_blue, replace=False)
    black_idx = set(picks[:n_black].tolist())
    blue_idx  = set(picks[n_black:].tolist())

    grey_xs, grey_ys = [], []
    black_xs, black_ys = [], []
    blue_xs,  blue_ys  = [], []
    for i in range(n):
        if i in black_idx:
            black_xs.append(xs[i]); black_ys.append(ys[i])
        elif i in blue_idx:
            blue_xs.append(xs[i]);  blue_ys.append(ys[i])
        else:
            grey_xs.append(xs[i]); grey_ys.append(ys[i])

    ax.scatter(grey_xs, grey_ys, s=DOT_S, color=DARK_GRAY,
                alpha=0.55, edgecolor="none", zorder=2)
    ax.scatter(black_xs, black_ys, s=DOT_S, color="#111",
                edgecolor="none", zorder=4)
    ax.scatter(blue_xs,  blue_ys,  s=DOT_S, color=ACCENT_BLUE,
                edgecolor="none", zorder=4)


def _glyph_composition(ax, cx, cy, w, h):
    """Mini diverging heatmap-strip (5 rows × 3 cols)."""
    n_rows = 5
    n_cols = 3
    cell_w = w / n_cols
    cell_h = h / n_rows
    rng = np.random.default_rng(7)
    vals = rng.normal(0.0, 0.6, (n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            v = vals[r, c]
            t = max(-1.0, min(1.0, v))
            if t >= 0:
                a = t
                col = (1.0 - a * (1.0 - 0x18 / 255),
                        1.0 - a * (1.0 - 0xA8 / 255),
                        1.0 - a * (1.0 - 0xE8 / 255))
            else:
                a = -t
                col = (1.0 - a * (1.0 - 0x9E / 255),
                        1.0 - a * (1.0 - 0x9E / 255),
                        1.0 - a * (1.0 - 0x9E / 255))
            x = cx - w / 2 + c * cell_w
            y = cy - h / 2 + r * cell_h
            ax.add_patch(Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=col, edgecolor="white", linewidth=0.4,
            ))


def _draw_deepcirc_cartoon(ax, cx, cy, w, h):
    """Three-section pictogram of the trained DeepCirc model:
        topology DAG  →  MLP  →  (c, g) scores.

    Section labels sit ABOVE each glyph (subtle grey italics), elements
    sit on a common centreline. Arrows are short and visually weighted so
    the eye reads left-to-right cleanly."""
    # Vertical anchors: labels at the top strip of `h`, element centres
    # on the cartoon's own vertical midline.
    label_y = cy - h / 2 + 0.5
    el_cy   = cy + 1.5

    # Allocate horizontal real estate. Three equal-ish sections separated
    # by two arrow strips. (Section ratios tuned so all three glyphs
    # have roughly the same visual weight after rendering.)
    sec_topo = 0.30 * w
    sec_mlp  = 0.32 * w
    sec_out  = 0.22 * w
    arrow_gap = (w - sec_topo - sec_mlp - sec_out) / 2

    x0_topo = cx - w / 2
    x1_topo = x0_topo + sec_topo
    x0_mlp  = x1_topo + arrow_gap
    x1_mlp  = x0_mlp + sec_mlp
    x0_out  = x1_mlp + arrow_gap
    x1_out  = x0_out + sec_out

    # --- Section labels (above each glyph) ----------------------------
    for x_c, txt in (((x0_topo + x1_topo) / 2, "Topology"),
                      ((x0_mlp + x1_mlp) / 2,   "MLP"),
                      ((x0_out + x1_out) / 2,   "Scores")):
        ax.text(x_c, label_y, txt,
                  ha="center", va="top",
                  fontsize=7, color=DARK_GRAY, style="italic")

    # --- Topology DAG (4-node mini-circuit, fan-out + fan-in) ---------
    # IN at left-centre, two internal regulators stacked, OUT at right.
    # Spaced for readability — wider than before so the diamond doesn't
    # look pinched.
    dag_pad_x = 1.2
    dag_pad_y = 4.0
    n_in  = (x0_topo + dag_pad_x,               el_cy)
    n_r1  = ((x0_topo + x1_topo) / 2,           el_cy - dag_pad_y)
    n_r2  = ((x0_topo + x1_topo) / 2,           el_cy + dag_pad_y)
    n_out = (x1_topo - dag_pad_x,               el_cy)
    NODE_R_DAG = 1.5
    edges = [(n_in, n_r1), (n_in, n_r2), (n_r1, n_out), (n_r2, n_out)]
    for (x0, y0), (x1, y1) in edges:
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-", mutation_scale=4,
            color=DARK_GRAY, linewidth=0.7,
            shrinkA=NODE_R_DAG * 72 / 25.4,  # mm radius → points
            shrinkB=NODE_R_DAG * 72 / 25.4,
        ))
    # Light grey for the three "non-output" nodes, dark grey for OUT
    # (matches figS11's ordinal palette).
    for nx, ny, is_out in ((n_in[0],  n_in[1],  False),
                             (n_r1[0], n_r1[1], False),
                             (n_r2[0], n_r2[1], False),
                             (n_out[0], n_out[1], True)):
        ax.add_patch(Circle((nx, ny), NODE_R_DAG,
                              facecolor=DARK_GRAY if is_out else LIGHT_GRAY,
                              edgecolor=DARK_GRAY, linewidth=0.5))

    # --- Arrow from topology → MLP (centred between VISUAL edges) -----
    # Section boundaries don't match visual content boundaries (DAG
    # has dag_pad_x slack, MLP has MLP_INSET on each side), so center
    # on the visual edges instead of the section midpoint.
    MLP_INSET_VIS = 3.0   # keep in sync with MLP_INSET below
    # Topology visual right edge = rightmost NODE EDGE, not centre
    # (NODE_R_DAG = 1.5 mm, dag_pad_x = 1.2 mm, so the rightmost node
    # actually extends 0.3 mm past x1_topo).
    topo_visual_r = x1_topo - dag_pad_x + NODE_R_DAG
    mlp_visual_l  = x0_mlp + MLP_INSET_VIS
    arrow_half = 2.2
    arrow_mid1 = (topo_visual_r + mlp_visual_l) / 2
    ax.add_patch(FancyArrowPatch(
        (arrow_mid1 - arrow_half, el_cy),
        (arrow_mid1 + arrow_half, el_cy),
        arrowstyle="-|>", mutation_scale=8,
        color=DARK_GRAY, linewidth=1.0,
    ))

    # --- MLP block: 3-layer × 3-node fully connected ------------------
    # Columns inset from the section boundary so the surrounding arrows
    # have proper visual breathing room instead of butting against the
    # outer dots.
    n_layers = 3
    n_nodes  = 3
    NODE_R_MLP = 0.95
    MLP_INSET  = 3.0
    col_xs = np.linspace(x0_mlp + MLP_INSET + NODE_R_MLP,
                          x1_mlp - MLP_INSET - NODE_R_MLP, n_layers)
    # Bumped vertical spread so the 3 layers read distinctly (was 5.2 mm
    # total — felt cramped with the connection mesh on top).
    row_ys = np.linspace(el_cy - 3.5, el_cy + 3.5, n_nodes)
    # Connection lines first — zorder > 1 to clear the left-box white
    # FancyBboxPatch background. Lightened from DARK_GRAY @ 0.85α to
    # MID_GRAY @ 0.7α so the network reads as a subtle web instead of
    # a heavy black mesh.
    for li in range(n_layers - 1):
        for r1 in row_ys:
            for r2 in row_ys:
                ax.plot([col_xs[li]    + NODE_R_MLP,
                          col_xs[li + 1] - NODE_R_MLP],
                         [r1, r2],
                         color=MID_GRAY, linewidth=0.7,
                         alpha=0.7, zorder=2.0,
                         solid_capstyle="round")
    for cm in col_xs:
        for ry in row_ys:
            ax.add_patch(Circle((cm, ry), NODE_R_MLP,
                                  facecolor="white",
                                  edgecolor=DARK_GRAY, linewidth=0.55))

    # --- Scores layout (compute first so arrow 2 can centre on the
    #     bars' visual left edge, not the section boundary).
    scores_cx = (x0_out + x1_out) / 2
    bar_w   = 1.7
    bar_gap = 1.2
    baseline_y = el_cy + 3.5
    h_circ = 5.2
    h_grow = 3.4
    bar1_x = scores_cx - (bar_w + bar_gap / 2)
    bar2_x = scores_cx + (bar_gap / 2)
    scores_visual_l = bar1_x

    # --- Arrow from MLP → scores (centred between VISUAL edges) -------
    mlp_visual_r = x1_mlp - MLP_INSET_VIS
    arrow_mid2 = (mlp_visual_r + scores_visual_l) / 2
    ax.add_patch(FancyArrowPatch(
        (arrow_mid2 - arrow_half, el_cy),
        (arrow_mid2 + arrow_half, el_cy),
        arrowstyle="-|>", mutation_scale=8,
        color=DARK_GRAY, linewidth=1.0,
    ))

    # --- Scores block: minimal 2-bar chart ---------------------------
    # Circuit bar — accent blue for emphasis.
    ax.add_patch(Rectangle(
        (bar1_x, baseline_y - h_circ),
        bar_w, h_circ,
        facecolor=ACCENT_BLUE, edgecolor="none", zorder=2,
    ))
    # Growth bar — paper mid-grey neutral.
    ax.add_patch(Rectangle(
        (bar2_x, baseline_y - h_grow),
        bar_w, h_grow,
        facecolor=MID_GRAY, edgecolor="none", zorder=2,
    ))
    # x-axis baseline only (no y-axis line).
    ax.plot(
        [bar1_x - 0.4, bar2_x + bar_w + 0.4],
        [baseline_y, baseline_y],
        color=DARK_GRAY, linewidth=0.7, solid_capstyle="round", zorder=3,
    )
    # Tilted italic labels below the baseline, one per bar.
    label_y = baseline_y + 0.7
    ax.text(bar1_x + bar_w / 2, label_y, "circuit",
              ha="right", va="top", rotation=35,
              rotation_mode="anchor",
              fontsize=6.0, color="#000000", style="italic")
    ax.text(bar2_x + bar_w / 2, label_y, "growth",
              ha="right", va="top", rotation=35,
              rotation_mode="anchor",
              fontsize=6.0, color="#000000", style="italic")


# --- panel assembly --------------------------------------------------------

def build_panel() -> None:
    use_style()

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    # Axes covers whole canvas; mm coords with top-left origin (y inverted).
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, CANVAS_W_MM)
    ax.set_ylim(CANVAS_H_MM, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ===========================================================
    # LEFT BOX — "Trained DeepCirc"
    # ===========================================================
    LEFT_X = 4.0
    LEFT_Y = 5.0
    LEFT_W = 72.0
    LEFT_H = 40.0

    ax.add_patch(FancyBboxPatch(
        (LEFT_X, LEFT_Y), LEFT_W, LEFT_H,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor="white", edgecolor=DARK_GRAY, linewidth=0.8,
    ))
    ax.text(LEFT_X + LEFT_W / 2, LEFT_Y + 5.0,
              "TRAINED DEEPCIRC",
              ha="center", va="center",
              fontsize=10, fontweight="bold", color="#000000")
    # Cartoon centred vertically in the lower 2/3 of the box (gives the
    # header room to breathe and keeps the model pictogram balanced now
    # that the subtitle line has been removed).
    _draw_deepcirc_cartoon(ax,
                            cx=LEFT_X + LEFT_W / 2,
                            cy=LEFT_Y + LEFT_H * 0.62,
                            w=LEFT_W - 6.0, h=22.0)

    # ===========================================================
    # 4 NUMBERED THUMBNAIL BOXES
    # No connecting arrows — the left-to-right reading order carries
    # the "input → lenses" implication without a visible trunk.
    # ===========================================================
    THUMB_X0  = 80.0
    THUMB_W   = 35.0   # 4 × 35 + 3 × 4 = 152 → fits exactly inside the
                       # 232 mm panel canvas (was 36 → overflowed 4 mm).
    THUMB_GAP = 4.0
    THUMB_Y   = 5.0
    THUMB_H   = 40.0

    LENSES = [
        ("1", "Population\nCartography",             _glyph_population),
        ("2", "Structural\nMotifs",                  _glyph_motif),
        ("3", "Cohort\nSelection",                   _glyph_cohort),
        ("4", "Composition $\\rightarrow$ Function", _glyph_composition),
    ]

    for i, (num, name, glyph_fn) in enumerate(LENSES):
        x = THUMB_X0 + i * (THUMB_W + THUMB_GAP)
        ax.add_patch(FancyBboxPatch(
            (x, THUMB_Y), THUMB_W, THUMB_H,
            boxstyle="round,pad=0,rounding_size=1.2",
            facecolor="white", edgecolor=BOX_BORDER, linewidth=0.7,
        ))
        # Numbered ACCENT_BLUE disc, top-left of box.
        num_cx = x + 4.0
        num_cy = THUMB_Y + 4.0
        ax.add_patch(Circle((num_cx, num_cy), 2.4,
                              facecolor=ACCENT_BLUE,
                              edgecolor="none"))
        ax.text(num_cx, num_cy, num,
                  ha="center", va="center",
                  fontsize=8, fontweight="bold", color="white")
        # Glyph + caption stack, centred vertically inside the box.
        # With fig-ref text dropped, glyph + caption have room to
        # breathe — glyph sits ~13 mm from top, caption at ~26 mm.
        glyph_fn(ax,
                  cx=x + THUMB_W / 2,
                  cy=THUMB_Y + 16.0,
                  w=THUMB_W - 8.0, h=15.0)
        ax.text(x + THUMB_W / 2, THUMB_Y + 28.5, name,
                  ha="center", va="top",
                  fontsize=8, color="#000000")

    out_base = PANELS_VEC / "panel_a"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel A written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
