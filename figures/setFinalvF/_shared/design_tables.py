"""Shared publication layout for the two curated design tables.

CSV files remain authoritative. Physical layout and measured text widths avoid
overlapping headers, clipped part sequences and inconsistent figure scaling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from PIL import Image

from figures.figtools import figsize_mm, use_style
from figures.styles.colors import (
    ACCENT_BLUE, BLACK, DARK_GRAY, LIGHT_GRAY, PASTEL, WHITE,
)

# Categorical identities from the delivered FigS13v2 / FigS15 figures.
TOP5_COLORS = {
    "PhlF/P2": "#F0B070",
    "QacR/Q2": "#E06070",
    "BetI/E1": "#90B5D0",
    "AmtR/A1": "#90C080",
    "SrpR/S4": "#5080C0",
}
WIDTH_MM = 230.0
ROW_MM = 5.8
BODY_TOP_MM = 17.0
BODY_PT = 9.0


class TableCanvas:
    """Draw in millimeters and reject text outside its allotted table cell."""

    def __init__(self, height):
        use_style()
        # Require Arial rather than silently substituting a sans-serif fallback.
        findfont(FontProperties(family="Arial"), fallback_to_default=False)
        plt.rcParams["font.family"] = ["Arial"]
        self.height = height
        self.fig = plt.figure(figsize=figsize_mm(WIDTH_MM, height), dpi=144)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, WIDTH_MM), ylim=(height, 0))
        self.ax.set_axis_off()
        self.fig.canvas.draw()
        self.text_cells = []

    def text(self, x, y, value, *, size=BODY_PT, color=BLACK,
             weight="normal", ha="left", cell=None):
        artist = self.ax.text(
            x, y, str(value), fontsize=size, color=color, fontfamily="Arial",
            fontweight=weight, ha=ha, va="center", linespacing=1.25,
        )
        self.text_cells.append((artist, cell))
        return artist

    def line(self, y, *, color=LIGHT_GRAY, width=0.35):
        self.ax.plot([6, 224], [y, y], color=color, lw=width,
                     solid_capstyle="butt", zorder=1)

    def header(self, columns, top, bottom):
        self.ax.add_patch(Rectangle(
            (6, top), 218, bottom - top, facecolor=PASTEL["blue"],
            alpha=0.16, edgecolor="none", zorder=0,
        ))
        self.line(top, color=ACCENT_BLUE, width=0.8)
        self.line(bottom, color=DARK_GRAY, width=0.55)
        for key, label, x, ha, left, right in columns:
            self.text(x, (top + bottom) / 2, label, size=8.5,
                      color=DARK_GRAY, weight="bold", ha=ha,
                      cell=(left, top + 0.8, right, bottom - 0.8))

    def part_sequence(self, x, y, value, right, top, bottom):
        """Advance each colored token by its actual rendered width."""
        renderer = self.fig.canvas.get_renderer()
        for i, part in enumerate(str(value).split(" → ")):
            if i:
                arrow = self.text(x, y, " → ", color=DARK_GRAY,
                                  cell=(x, top, right, bottom))
                x += arrow.get_window_extent(renderer).width * 25.4 / self.fig.dpi
            artist = self.text(
                x, y, part, color=TOP5_COLORS.get(part, BLACK),
                weight="bold" if part in TOP5_COLORS else "normal",
                cell=(x, top, right, bottom),
            )
            x += artist.get_window_extent(renderer).width * 25.4 / self.fig.dpi

    def export(self, out_base):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        for artist, cell in self.text_cells:
            box = artist.get_window_extent(renderer).transformed(inv)
            left, right = sorted((box.x0, box.x1))
            top, bottom = sorted((box.y0, box.y1))
            limits = cell or (0.5, 0.5, WIDTH_MM - 0.5, self.height - 0.5)
            if (left < limits[0] - 0.15 or top < limits[1] - 0.15
                    or right > limits[2] + 0.15 or bottom > limits[3] + 0.15):
                raise ValueError(f"Text exceeds its table cell: {artist.get_text()!r}")

        # Override panel defaults: fixed canvas, editable text, opaque white.
        with plt.rc_context({"savefig.bbox": None, "savefig.transparent": False,
                             "svg.fonttype": "none"}):
            for ext in ("pdf", "svg", "png"):
                path = out_base.with_suffix(f".{ext}")
                self.fig.savefig(path, dpi=600, bbox_inches=None, pad_inches=0,
                                 facecolor=WHITE, transparent=False)
                if ext == "png":
                    with Image.open(path) as image:
                        image.convert("RGB").save(path, dpi=(600, 600))
                print(f"Wrote: {path}")
        plt.close(self.fig)


def _columns(include_combined):
    columns = [
        ("rank", "Rank", 6, "left", 6, 16),
        ("target", "Target\nfunction", 19, "left", 19, 34),
        ("size", "Size", 43, "center", 36, 50),
        ("circuit_score", "Circuit\nscore", 68, "right", 53, 70),
        ("growth_score", "Growth\nscore", 89, "right", 74, 91),
    ]
    if include_combined:
        columns.append(("combined_score", "Combined\nscore", 114, "right", 96, 116))
    parts_left = 120 if include_combined else 98
    parts_label = "Part names of circuit components"
    columns.append(("part_names", parts_label,
                    parts_left, "left", parts_left, 224))
    return columns


def _format(key, value):
    if key == "rank":
        return str(int(value))
    decimals = {"circuit_score": 2, "growth_score": 3, "combined_score": 4}
    if key in decimals:
        return f"{float(value):,.{decimals[key]}f}"
    return str(value)


def render_design_table(rows, out_base: Path, *, number: int, footer=None):
    """Render saved values and row order without recomputing the analysis."""
    include_combined = number == 2
    body_end = BODY_TOP_MM + len(rows) * ROW_MM
    # Only headers and data belong in the artwork. Titles, definitions and
    # color explanations live separately in ../table_legends.md.
    canvas = TableCanvas(body_end + (53 if include_combined else 6))

    columns = _columns(include_combined)
    canvas.header(columns, 6, BODY_TOP_MM)
    for i, row in enumerate(rows):
        top = BODY_TOP_MM + i * ROW_MM
        bottom = top + ROW_MM
        y = (top + bottom) / 2
        for key, label, x, ha, left, right in columns:
            if key == "part_names":
                canvas.part_sequence(x, y, row[key], right, top, bottom)
            else:
                canvas.text(x, y, _format(key, row[key]), ha=ha,
                            color=DARK_GRAY if key == "rank" else BLACK,
                            cell=(left, top, right, bottom))
        canvas.line(bottom)
    canvas.line(body_end, color=DARK_GRAY, width=0.55)

    if include_combined:
        _draw_frequency_table(canvas, footer, body_end)
    canvas.export(out_base)


def _draw_frequency_table(canvas, rows, body_end):
    columns = [
        ("rank", "Rank", 6, "left", 6, 16),
        ("part_name", "Part", 23, "left", 23, 58),
        ("count_in_top15_maxcirc", "Table S1", 113, "right", 76, 115),
        ("count_in_top15_knee", "Table S2", 168, "right", 129, 170),
        ("total_count_out_of_30", "Total", 221, "right", 187, 224),
    ]
    top = body_end + 8
    body_top = top + 10
    canvas.header(columns, top, body_top)
    for i, row in enumerate(rows):
        y_top = body_top + i * ROW_MM
        y_bottom = y_top + ROW_MM
        for key, label, x, ha, left, right in columns:
            is_part = key == "part_name"
            canvas.text(x, (y_top + y_bottom) / 2, row[key], ha=ha,
                        color=TOP5_COLORS[row[key]] if is_part else BLACK,
                        weight="bold" if is_part else "normal",
                        cell=(left, y_top, right, y_bottom))
        canvas.line(y_bottom)
    canvas.line(body_top + len(rows) * ROW_MM, color=DARK_GRAY, width=0.55)
