"""SVG-based schematic helpers for circuit cartoons, workflow boxes, icons.

Each helper writes a self-contained `.svg` file using `svgwrite`. SVG output
is the canonical form so the components stay vector-editable in Illustrator.

All shapes use muted pastel fills + thin DARK_GRAY strokes to match the
paper's schematic language. Components are positioned in a unitless
coordinate system; the caller chooses the canvas size in pixels (or
millimeters via the `size_mm` argument).

If `svgwrite` is not installed, the functions raise an informative
ImportError rather than failing silently. Install via:
    pip install svgwrite

Optional: install `cairosvg` to also render the SVGs to PNG/PDF inside
the same script — but the default behavior is SVG-only; PNG conversion
is intentionally a separate step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..styles.colors import (
    PASTEL, DARK_GRAY, MID_GRAY, LIGHT_GRAY, VERY_LIGHT_GRAY,
    ACCENT_BLUE, ACCENT_BLUE_LIGHT,
)
from ..styles.typography import (
    DEFAULT_FONT_FAMILY_STR, ANNOTATION_SIZE, AXIS_LABEL_SIZE,
)


def _require_svgwrite():
    try:
        import svgwrite  # noqa: F401
        return svgwrite
    except ImportError as e:
        raise ImportError(
            "svgwrite is required for figures.figtools.schematics. "
            "Install via `pip install svgwrite` (or run "
            "`pip install -r requirements.txt`)."
        ) from e


# ---------------------------------------------------------------------------
# Drawing primitives — each returns the SVG element (so callers can compose)
# ---------------------------------------------------------------------------

def draw_gene_arrow(
    dwg,
    *,
    x: float,
    y: float,
    width: float = 40.0,
    height: float = 10.0,
    fill: str = PASTEL["green"],
    stroke: str = DARK_GRAY,
    stroke_width: float = 0.6,
    direction: str = "right",
):
    """An arrow-shaped gene body. Anchored at (x, y) at the tail."""
    head_w = height * 1.2
    if direction == "right":
        pts = [
            (x, y),
            (x + width - head_w, y),
            (x + width - head_w, y - height * 0.4),
            (x + width, y + height * 0.5),
            (x + width - head_w, y + height * 1.4),
            (x + width - head_w, y + height),
            (x, y + height),
        ]
    else:
        pts = [
            (x + width, y),
            (x + head_w, y),
            (x + head_w, y - height * 0.4),
            (x, y + height * 0.5),
            (x + head_w, y + height * 1.4),
            (x + head_w, y + height),
            (x + width, y + height),
        ]
    return dwg.polygon(
        points=pts, fill=fill, stroke=stroke, stroke_width=stroke_width,
        stroke_linejoin="round",
    )


def draw_promoter(
    dwg,
    *,
    x: float,
    y: float,
    width: float = 12.0,
    height: float = 10.0,
    stroke: str = DARK_GRAY,
    stroke_width: float = 0.8,
):
    """A promoter glyph — bent arrow indicating transcription start."""
    grp = dwg.g(stroke=stroke, stroke_width=stroke_width, fill="none",
                stroke_linecap="round", stroke_linejoin="round")
    # vertical riser + horizontal arrow
    grp.add(dwg.line(start=(x, y + height), end=(x, y + 1)))
    grp.add(dwg.line(start=(x, y + 1), end=(x + width - 2, y + 1)))
    grp.add(dwg.polyline(points=[
        (x + width - 4, y - 1.5),
        (x + width, y + 1),
        (x + width - 4, y + 3.5),
    ]))
    return grp


def draw_terminator(
    dwg,
    *,
    x: float,
    y: float,
    width: float = 6.0,
    height: float = 10.0,
    stroke: str = DARK_GRAY,
    stroke_width: float = 0.8,
):
    """A terminator glyph — vertical bar with horizontal cap, hairpin-like."""
    grp = dwg.g(stroke=stroke, stroke_width=stroke_width, fill="none",
                stroke_linecap="round")
    grp.add(dwg.line(start=(x + width / 2, y + height), end=(x + width / 2, y)))
    grp.add(dwg.line(start=(x, y), end=(x + width, y)))
    return grp


def draw_repressor_arc(
    dwg,
    *,
    x0: float, y0: float,
    x1: float, y1: float,
    arc_height: float = 12.0,
    stroke: str = MID_GRAY,
    stroke_width: float = 0.6,
):
    """A T-bar repression arc from (x0, y0) to (x1, y1)."""
    grp = dwg.g(stroke=stroke, stroke_width=stroke_width, fill="none",
                stroke_linecap="round")
    midx = (x0 + x1) / 2.0
    midy = min(y0, y1) - arc_height
    path_d = (
        f"M {x0:.2f} {y0:.2f} "
        f"Q {midx:.2f} {midy:.2f}, {x1:.2f} {y1:.2f}"
    )
    grp.add(dwg.path(d=path_d))
    # T-cap at target end
    cap_half = 3.0
    grp.add(dwg.line(start=(x1 - cap_half, y1), end=(x1 + cap_half, y1)))
    return grp


def draw_input_label(
    dwg,
    *,
    x: float, y: float,
    text: str,
    fill: str = ACCENT_BLUE,
    text_color: str = "white",
    radius: float = 6.0,
    font_size: float = ANNOTATION_SIZE,
):
    """Filled circle with a short label inside — used for sensor inputs."""
    grp = dwg.g()
    grp.add(dwg.circle(center=(x, y), r=radius,
                       fill=fill, stroke=DARK_GRAY, stroke_width=0.4))
    grp.add(dwg.text(
        text, insert=(x, y + font_size * 0.35),
        text_anchor="middle",
        fill=text_color,
        font_family=DEFAULT_FONT_FAMILY_STR,
        font_size=f"{font_size}pt",
        font_weight="bold",
    ))
    return grp


def draw_output_label(
    dwg,
    *,
    x: float, y: float,
    text: str,
    fill: str = DARK_GRAY,
    radius: float = 6.0,
    font_size: float = ANNOTATION_SIZE,
):
    """Outlined circle for an output node."""
    grp = dwg.g()
    grp.add(dwg.circle(center=(x, y), r=radius,
                       fill="white", stroke=fill, stroke_width=0.6))
    grp.add(dwg.text(
        text, insert=(x, y + font_size * 0.35),
        text_anchor="middle",
        fill=fill,
        font_family=DEFAULT_FONT_FAMILY_STR,
        font_size=f"{font_size}pt",
        font_weight="bold",
    ))
    return grp


def draw_workflow_box(
    dwg,
    *,
    x: float, y: float,
    width: float = 70.0, height: float = 28.0,
    title: str = "",
    subtitle: str | None = None,
    fill: str = VERY_LIGHT_GRAY,
    stroke: str = DARK_GRAY,
    stroke_width: float = 0.6,
):
    """A rounded rectangle workflow box with a title and optional subtitle."""
    grp = dwg.g()
    grp.add(dwg.rect(
        insert=(x, y), size=(width, height),
        rx=3, ry=3,
        fill=fill, stroke=stroke, stroke_width=stroke_width,
    ))
    grp.add(dwg.text(
        title, insert=(x + width / 2, y + height / 2 + 1),
        text_anchor="middle",
        fill=DARK_GRAY,
        font_family=DEFAULT_FONT_FAMILY_STR,
        font_size=f"{AXIS_LABEL_SIZE}pt",
        font_weight="bold",
    ))
    if subtitle:
        grp.add(dwg.text(
            subtitle, insert=(x + width / 2, y + height / 2 + 9),
            text_anchor="middle",
            fill=MID_GRAY,
            font_family=DEFAULT_FONT_FAMILY_STR,
            font_size=f"{ANNOTATION_SIZE}pt",
        ))
    return grp


def draw_arrow_between_boxes(
    dwg,
    *,
    x0: float, y0: float,
    x1: float, y1: float,
    stroke: str = MID_GRAY,
    stroke_width: float = 0.8,
):
    """Straight arrow between two points, with arrow head at the destination."""
    grp = dwg.g(stroke=stroke, stroke_width=stroke_width, fill="none",
                stroke_linecap="round", stroke_linejoin="round")
    grp.add(dwg.line(start=(x0, y0), end=(x1, y1)))
    # arrow head
    dx, dy = (x1 - x0), (y1 - y0)
    length = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    # perpendicular
    px, py = -uy, ux
    head_len = 4.0
    head_w = 2.0
    base_x = x1 - ux * head_len
    base_y = y1 - uy * head_len
    head_pts = [
        (x1, y1),
        (base_x + px * head_w, base_y + py * head_w),
        (base_x - px * head_w, base_y - py * head_w),
    ]
    grp.add(dwg.polygon(points=head_pts, fill=stroke, stroke="none"))
    return grp


def draw_neural_network_icon(
    dwg,
    *,
    x: float, y: float,
    layer_sizes: Sequence[int] = (3, 4, 4, 2),
    node_radius: float = 2.2,
    horiz_gap: float = 14.0,
    vert_gap: float = 6.0,
    node_fill: str = ACCENT_BLUE,
    edge_stroke: str = LIGHT_GRAY,
):
    """Mini neural-network icon. (x, y) is the top-left of the icon bbox."""
    grp = dwg.g()
    nodes_per_layer: list[list[tuple[float, float]]] = []
    for li, size in enumerate(layer_sizes):
        col_x = x + li * horiz_gap
        col_height = (size - 1) * vert_gap
        col_top = y + (max(layer_sizes) - 1) * vert_gap / 2 - col_height / 2
        nodes_per_layer.append([
            (col_x, col_top + ni * vert_gap) for ni in range(size)
        ])
    # edges
    for li in range(len(nodes_per_layer) - 1):
        for n0 in nodes_per_layer[li]:
            for n1 in nodes_per_layer[li + 1]:
                grp.add(dwg.line(
                    start=n0, end=n1,
                    stroke=edge_stroke, stroke_width=0.3,
                ))
    # nodes (on top)
    for layer in nodes_per_layer:
        for nx, ny in layer:
            grp.add(dwg.circle(
                center=(nx, ny), r=node_radius,
                fill=node_fill, stroke=DARK_GRAY, stroke_width=0.3,
            ))
    return grp


def draw_design_space_icon(
    dwg,
    *,
    x: float, y: float,
    width: float = 40.0,
    height: float = 30.0,
    n_points: int = 60,
    highlight_n: int = 3,
    seed: int = 0,
):
    """Tiny pseudo-scatter icon representing a design space."""
    import random
    rng = random.Random(seed)
    grp = dwg.g()
    grp.add(dwg.rect(
        insert=(x, y), size=(width, height),
        rx=2, ry=2,
        fill=VERY_LIGHT_GRAY, stroke=LIGHT_GRAY, stroke_width=0.4,
    ))
    pts: list[tuple[float, float]] = []
    for _ in range(n_points):
        px = x + rng.uniform(2, width - 2)
        py = y + rng.uniform(2, height - 2)
        pts.append((px, py))
        grp.add(dwg.circle(center=(px, py), r=0.8,
                           fill=MID_GRAY, stroke="none", opacity=0.7))
    # highlight a small number of points
    for px, py in rng.sample(pts, k=min(highlight_n, len(pts))):
        grp.add(dwg.circle(center=(px, py), r=1.6,
                           fill=ACCENT_BLUE,
                           stroke=DARK_GRAY, stroke_width=0.3))
    return grp


def draw_circuit_cartoon(
    dwg,
    *,
    x: float, y: float,
    inputs: Sequence[str] = ("IPTG", "aTc", "Ara"),
    output: str = "YFP",
    width: float = 110.0,
    height: float = 60.0,
):
    """A compact 3-input → output circuit cartoon, paper schematic style."""
    grp = dwg.g()
    # backdrop
    grp.add(dwg.rect(insert=(x, y), size=(width, height), rx=3, ry=3,
                     fill="white", stroke=LIGHT_GRAY, stroke_width=0.4))

    # input nodes on left, evenly spaced
    in_x = x + 10
    spacing = (height - 16) / max(len(inputs) - 1, 1)
    for i, name in enumerate(inputs):
        cy = y + 8 + i * spacing
        grp.add(draw_input_label(dwg, x=in_x, y=cy, text=name[:3], radius=5))

    # two internal gates as small pastel circles
    g1x, g1y = x + width / 2 - 12, y + height / 2 - 8
    g2x, g2y = x + width / 2 + 4, y + height / 2 + 6
    for cx, cy, hue in [(g1x, g1y, "teal"), (g2x, g2y, "purple")]:
        grp.add(dwg.circle(center=(cx, cy), r=4.5,
                           fill=PASTEL[hue],
                           stroke=DARK_GRAY, stroke_width=0.4))

    # output node on right
    out_x = x + width - 12
    out_y = y + height / 2
    grp.add(draw_output_label(dwg, x=out_x, y=out_y, text=output))

    # connecting arrows: inputs -> g1, inputs -> g2 (alternating), gates -> out
    for i in range(len(inputs)):
        cy = y + 8 + i * spacing
        target = (g1x - 5, g1y) if i % 2 == 0 else (g2x - 5, g2y)
        grp.add(draw_arrow_between_boxes(dwg, x0=in_x + 5, y0=cy,
                                          x1=target[0], y1=target[1]))
    grp.add(draw_arrow_between_boxes(
        dwg, x0=g1x + 5, y0=g1y, x1=out_x - 6, y1=out_y - 2))
    grp.add(draw_arrow_between_boxes(
        dwg, x0=g2x + 5, y0=g2y, x1=out_x - 6, y1=out_y + 2))
    return grp


# ---------------------------------------------------------------------------
# Top-level convenience — build a standalone SVG from scratch.
# ---------------------------------------------------------------------------

def new_canvas(out_path: str | Path, *, size_mm: tuple[float, float] = (60, 35)):
    """Create a fresh SVG canvas sized in mm, returning the Drawing.

    Caller adds shapes via the draw_* helpers, then calls `dwg.save()`.

    >>> dwg = new_canvas("workflow.svg", size_mm=(80, 30))
    >>> dwg.add(draw_workflow_box(dwg, x=5, y=5, title="Topology design"))
    >>> dwg.save()
    """
    svgwrite = _require_svgwrite()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_mm, h_mm = size_mm
    dwg = svgwrite.Drawing(
        str(out_path),
        size=(f"{w_mm}mm", f"{h_mm}mm"),
        viewBox=f"0 0 {w_mm} {h_mm}",
    )
    return dwg


__all__ = [
    "new_canvas",
    "draw_gene_arrow", "draw_promoter", "draw_terminator", "draw_repressor_arc",
    "draw_input_label", "draw_output_label",
    "draw_workflow_box", "draw_arrow_between_boxes",
    "draw_neural_network_icon", "draw_design_space_icon",
    "draw_circuit_cartoon",
]
