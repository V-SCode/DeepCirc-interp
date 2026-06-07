"""Figure S05 — Panel D: PLACEHOLDER (slot reserved, content TBD).

The previous Panel D content (cross-topology interaction conservation
heatmap, port of `deepcirc_sae/outputs/figures/presentation/
E4_interaction_conservation`) is preserved at
`panels/vector/panel_dv0.{pdf,svg}` + `panels/raster/panel_dv0.png` and
the build script at `scripts/build_panel_dv0.py`. A new Panel D figure
may be brought in later; this placeholder reserves the manifest slot
until then.

Outputs (vector + raster):
  panels/vector/panel_d.pdf | .svg
  panels/raster/panel_d.png

Run:
    python \\
        figures/setFinal/figS15/scripts/build_panel_d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import MID_GRAY  # noqa: E402


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS15"
              / "panels" / "vector")

TEXT_GREY = "#666666"

# Match the manifest slot for Panel D (220 × 150 mm).
CANVAS_W_MM = 220
CANVAS_H_MM = 150


def build_panel() -> None:
    use_style()

    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pad = 0.02
    border = Rectangle(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad,
        linewidth=0.8, edgecolor=MID_GRAY,
        facecolor="none", linestyle=(0, (4, 3)),
    )
    ax.add_patch(border)

    ax.text(0.5, 0.56, "panel d (placeholder)",
             ha="center", va="center",
             fontsize=10.0, fontweight="normal", color=TEXT_GREY)
    ax.text(0.5, 0.46, "content TBD",
             ha="center", va="center",
             fontsize=8.0, fontweight="normal", color=TEXT_GREY,
             style="italic")

    out_base = PANELS_VEC / "panel_d"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel D (placeholder) written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
