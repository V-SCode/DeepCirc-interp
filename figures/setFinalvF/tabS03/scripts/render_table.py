"""Render Supplementary Table S3 in the setFinal paper theme.

Reads the existing CSV without changing its values or row order. Cell color
encodes mean log2 enrichment on an explicit, zero-centered [-3, +3] scale;
values outside this range retain their printed value and saturate in color.
Outputs editable PDF/SVG and an opaque, white-background 600 dpi RGB PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT = SCRIPT_PATH.parents[4]
REPO_ROOT = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools import figsize_mm, use_style  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    ACCENT_BLUE_LIGHT, DARK_GRAY, LIGHT_GRAY,
    VERY_LIGHT_GRAY, WHITE,
)

BUCKETS = [
    "NOT-early", "NOT-middle", "NOT-late",
    "NOR-early", "NOR-middle", "NOR-late",
]

CANVAS_W_MM = 180.0
MARGIN_MM = 6.0
LABEL_W_MM = 29.0
GUTTER_MM = 3.0
CELL_W_MM = (CANVAS_W_MM - 2 * MARGIN_MM - LABEL_W_MM - GUTTER_MM) / 6
ROW_HEIGHT_MM = 5.5
HEADER_TOP_MM = 6.0
BODY_TOP_MM = HEADER_TOP_MM + 14.0
COLOR_LIMIT = 3.0


def _cell_left(column: int) -> float:
    return MARGIN_MM + LABEL_W_MM + column * CELL_W_MM + (GUTTER_MM if column >= 3 else 0)


def _make_cmap() -> mcolors.LinearSegmentedColormap:
    # Pale versions of the paper's gray/blue ramp keep all numerals legible.
    return mcolors.LinearSegmentedColormap.from_list(
        "paper_pale_gray_blue", [LIGHT_GRAY, WHITE, ACCENT_BLUE_LIGHT], N=257
    )


def build_table() -> None:
    use_style()
    out_dir = SCRIPT_PATH.parents[1] / "final"
    df = pd.read_csv(out_dir / "table_S3.csv")
    parts = df["part_name"].tolist()
    values = df[BUCKETS].to_numpy(dtype=float)
    body_bottom = BODY_TOP_MM + len(parts) * ROW_HEIGHT_MM
    canvas_h = body_bottom + MARGIN_MM
    cmap = _make_cmap()
    norm = mcolors.Normalize(vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, clip=True)

    # The figure stylesheet normally crops/exports transparent panels. Tables
    # need an exact artboard, white canvas, and editable vector typography.
    with mpl.rc_context({
        "font.family": "Arial", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": None, "savefig.pad_inches": 0,
        "savefig.transparent": False, "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }):
        fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, canvas_h))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, CANVAS_W_MM), ylim=(canvas_h, 0))
        ax.set_axis_off()

        def text(x: float, y: float, label: str, *, size: float = 8.5,
                 ha: str = "left", weight: str = "normal", color: str = DARK_GRAY):
            return ax.text(x, y, label, fontsize=size, ha=ha, va="center",
                           color=color, fontfamily="Arial", fontweight=weight)

        def rule(y: float, left: float = MARGIN_MM,
                 right: float = CANVAS_W_MM - MARGIN_MM, *,
                 color: str = LIGHT_GRAY, width: float = 0.45):
            ax.plot([left, right], [y, y], color=color, lw=width,
                    solid_capstyle="butt", zorder=4)

        # Group labels make the gate type/position hierarchy visible once.
        ax.add_patch(Rectangle((MARGIN_MM, HEADER_TOP_MM), LABEL_W_MM, 14,
                               facecolor=VERY_LIGHT_GRAY, edgecolor="none"))
        text(MARGIN_MM + 2, HEADER_TOP_MM + 7, "Part", weight="bold")
        for group_index, gate in enumerate(("NOT", "NOR")):
            left = _cell_left(group_index * 3)
            width = 3 * CELL_W_MM
            ax.add_patch(Rectangle((left, HEADER_TOP_MM), width, 14,
                                   facecolor=VERY_LIGHT_GRAY, edgecolor="none"))
            text(left + width / 2, HEADER_TOP_MM + 3.5, gate, ha="center", weight="bold")
            rule(HEADER_TOP_MM + 7, left + 2, left + width - 2, width=0.35)
            for offset, position in enumerate(("Early", "Middle", "Late")):
                text(left + (offset + 0.5) * CELL_W_MM, HEADER_TOP_MM + 10.5,
                     position, ha="center", size=8)
        rule(HEADER_TOP_MM, color=LIGHT_GRAY, width=0.6)
        rule(BODY_TOP_MM, color=LIGHT_GRAY, width=0.6)

        for row, part in enumerate(parts):
            top = BODY_TOP_MM + row * ROW_HEIGHT_MM
            center = top + ROW_HEIGHT_MM / 2
            # Labels have one color: only cell fill carries quantitative meaning.
            text(MARGIN_MM + 2, center, part)
            rule(top + ROW_HEIGHT_MM, MARGIN_MM, _cell_left(0), width=0.3)
            for column, value in enumerate(values[row]):
                left = _cell_left(column)
                missing = not np.isfinite(value)
                ax.add_patch(Rectangle(
                    (left, top), CELL_W_MM, ROW_HEIGHT_MM,
                    facecolor=VERY_LIGHT_GRAY if missing else cmap(norm(value)),
                    edgecolor=WHITE, linewidth=0.45,
                ))
                text(left + CELL_W_MM / 2, center,
                     "—" if missing else f"{value:+.2f}", ha="center")
        rule(body_bottom, color=LIGHT_GRAY, width=0.6)

        for suffix in ("pdf", "svg", "png"):
            output = out_dir / f"table_S3.{suffix}"
            fig.savefig(output, dpi=600, bbox_inches=None, pad_inches=0,
                        facecolor=WHITE, transparent=False)
            if suffix == "png":
                with Image.open(output) as rendered:
                    rgb = rendered.convert("RGB")
                rgb.save(output, dpi=(600, 600))
            print(f"Wrote: {output.relative_to(REPO_ROOT)}")
        plt.close(fig)


if __name__ == "__main__":
    build_table()
