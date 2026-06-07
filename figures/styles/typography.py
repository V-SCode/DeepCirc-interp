"""Typography constants + helpers for DeepCirc paper figures.

Sizes are in points. Panel labels are lowercase bold letters (a, b, c, ...)
placed at the top-left of each panel. The mplstyle file owns the
matplotlib-level rcParams; this module is for code that needs to reference
the same sizes outside of rcParams (e.g., placing text on a figure
canvas, drawing labels into SVG schematics).
"""
from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# Font sizes (points)
# ---------------------------------------------------------------------------

PANEL_LABEL_SIZE: float = 9.0   # lowercase bold panel letters (a, b, c, ...)
AXIS_LABEL_SIZE:  float = 7.0
TITLE_SIZE:       float = 7.0
TICK_LABEL_SIZE:  float = 6.0
LEGEND_SIZE:      float = 6.0
ANNOTATION_SIZE:  float = 6.0
CAPTION_SIZE:     float = 6.0

# Font stack — matches the mplstyle. Use these names anywhere matplotlib's
# rcParams don't reach (svgwrite text, manual fig.text calls, etc.).
DEFAULT_FONT_FAMILY: tuple[str, ...] = ("Arial", "Helvetica", "DejaVu Sans")
DEFAULT_FONT_FAMILY_STR: str = ", ".join(DEFAULT_FONT_FAMILY)

# Panel labels — lowercase. Extend by slicing more letters as needed.
PANEL_LETTERS: list[str] = list("abcdefghijklmnopqrstuvwxyz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_panel_label(
    fig,
    label: str,
    x: float,
    y: float,
    *,
    size: float = PANEL_LABEL_SIZE,
    weight: str = "bold",
    ha: str = "left",
    va: str = "top",
    color: str | None = None,
) -> None:
    """Draw a lowercase bold panel label on a matplotlib Figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    label : str
        Single character (or short string) — will be cast to lowercase.
    x, y : float
        Position in figure-fraction coordinates [0, 1].
    size : float
        Font size in points. Defaults to PANEL_LABEL_SIZE.
    weight : str
        Matplotlib font weight string.
    ha, va : str
        Horizontal / vertical alignment.
    color : str or None
        Defaults to dark gray (matches the spine color in the style file).
    """
    from .colors import DARK_GRAY  # local import to avoid circulars at import time

    fig.text(
        x, y, str(label).lower(),
        fontsize=size,
        fontweight=weight,
        ha=ha, va=va,
        color=color or DARK_GRAY,
        family=DEFAULT_FONT_FAMILY[0],
    )


def add_axes_panel_label(
    ax,
    label: str,
    *,
    x: float = -0.18,
    y: float = 1.05,
    size: float = PANEL_LABEL_SIZE,
    weight: str = "bold",
) -> None:
    """Same as add_panel_label but anchored to an axes in axes-fraction coords."""
    from .colors import DARK_GRAY

    ax.text(
        x, y, str(label).lower(),
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        ha="left", va="bottom",
        color=DARK_GRAY,
        family=DEFAULT_FONT_FAMILY[0],
    )


def panel_letters(n: int) -> list[str]:
    """Return the first n lowercase panel letters."""
    if n > len(PANEL_LETTERS):
        raise ValueError(f"Only {len(PANEL_LETTERS)} panel letters defined")
    return PANEL_LETTERS[:n]


__all__ = [
    "PANEL_LABEL_SIZE", "AXIS_LABEL_SIZE", "TITLE_SIZE",
    "TICK_LABEL_SIZE", "LEGEND_SIZE", "ANNOTATION_SIZE", "CAPTION_SIZE",
    "DEFAULT_FONT_FAMILY", "DEFAULT_FONT_FAMILY_STR",
    "PANEL_LETTERS",
    "add_panel_label", "add_axes_panel_label", "panel_letters",
]
