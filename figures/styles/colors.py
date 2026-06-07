"""DeepCirc paper figure palette + axis-style helpers.

Convention:
- ACCENT_BLUE is the single highlight color for active/quantitative bars,
  lines, and "this is the data we want you to look at" markers.
- ACCENT_BLUE_LIGHT is the desaturated companion for confidence bands,
  shaded regions, and secondary lines.
- LIGHT_GRAY / VERY_LIGHT_GRAY are the background / inactive-bar colors.
- DARK_GRAY is the default text + spine + stroke color.
- PASTEL contains muted hues for schematic circuit components (gene
  arrows, promoters, regulator badges, etc.). They are intentionally
  desaturated so they read as illustration, not data.

Keep this module data-free. Add new colors here; don't redefine them
in plotting scripts.
"""
from __future__ import annotations

from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# Core palette
# ---------------------------------------------------------------------------

ACCENT_BLUE       = "#18A8E8"  # highlight / active data (cyan bars, points)
ACCENT_BLUE_LIGHT = "#8DD4F6"  # confidence band / secondary
ACCENT_BLUE_DARK  = "#0C7AB0"  # outlined accents / strong emphasis

# Yellow dot — the paper's convention for "this is the top-selected design"
# on every design-space scatter (Fig 3b, repeated in Extended Data).
ACCENT_YELLOW     = "#FFD43A"  # top-pick markers
ACCENT_YELLOW_DARK = "#C99800"  # outline for yellow dots

DARK_GRAY         = "#4A4A4A"  # text + spines + strokes
MID_GRAY          = "#9E9E9E"  # secondary text + neutral markers
LIGHT_GRAY        = "#D9D9D9"  # inactive bars / background categories
VERY_LIGHT_GRAY   = "#F2F2F2"  # axes shading / panel backdrops

BLACK             = "#000000"
WHITE             = "#FFFFFF"


# ---------------------------------------------------------------------------
# Muted pastels for schematic components
# ---------------------------------------------------------------------------

PASTEL: dict[str, str] = {
    "green":  "#BFD9AE",
    "teal":   "#9FD3CF",
    "orange": "#E6B37A",
    "pink":   "#E7A3A3",
    "red":    "#D77A7A",
    "purple": "#C5A1D5",
    "yellow": "#E7D58A",
    "blue":   "#9EC7E8",
}

# Stable ordering for cycling through pastel hues
PASTEL_ORDER: tuple[str, ...] = (
    "green", "teal", "blue", "purple", "pink", "orange", "yellow", "red",
)


# ---------------------------------------------------------------------------
# Optional: TF-family / part-class semantic colors (paper-aligned pastels).
# Use these when a panel needs consistent coloring for biological parts.
# ---------------------------------------------------------------------------

FAMILY_COLORS: dict[str, str] = {
    "AmeR":   PASTEL["green"],
    "AmtR":   PASTEL["teal"],
    "BM3R1":  PASTEL["blue"],
    "BetI":   PASTEL["purple"],
    "HlyIIR": PASTEL["pink"],
    "IcaRA":  PASTEL["red"],
    "LitR":   PASTEL["orange"],
    "LmrA":   PASTEL["yellow"],
    "PhlF":   "#A3CFA3",
    "PsrA":   "#C7B8E0",
    "QacR":   "#F0BFA0",
    "SrpR":   "#9FBEDB",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_part_color(name_or_index: str | int) -> str:
    """Return a stable color for a TF family / part name or an integer index.

    Strings: looks up FAMILY_COLORS exact match (case-insensitive), falls
    back to MID_GRAY for unknown parts.

    Integers: cycles through PASTEL_ORDER.
    """
    if isinstance(name_or_index, int):
        key = PASTEL_ORDER[name_or_index % len(PASTEL_ORDER)]
        return PASTEL[key]
    name = str(name_or_index)
    # exact match first
    if name in FAMILY_COLORS:
        return FAMILY_COLORS[name]
    # case-insensitive fallback (handles "phlf" / "PHLF")
    lower = {k.lower(): v for k, v in FAMILY_COLORS.items()}
    if name.lower() in lower:
        return lower[name.lower()]
    return MID_GRAY


def apply_axis_style(ax) -> None:
    """Apply the locked spine/tick aesthetic to a matplotlib Axes.

    Use this on axes built outside the plots.* helpers (e.g., inset axes,
    custom panels) so they match the paper style.
    """
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK_GRAY)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major",
                   color=DARK_GRAY, width=0.6, length=2.5,
                   labelcolor=DARK_GRAY)
    ax.xaxis.label.set_color(DARK_GRAY)
    ax.yaxis.label.set_color(DARK_GRAY)


def style_barplot(ax) -> None:
    """Cosmetic pass for bar plots — apply after `ax.bar(...)`."""
    apply_axis_style(ax)
    ax.margins(x=0.02)
    ax.grid(False)


def style_scatter(ax) -> None:
    """Cosmetic pass for scatter plots — apply after `ax.scatter(...)`."""
    apply_axis_style(ax)
    ax.grid(False)


def style_lineplot(ax) -> None:
    """Cosmetic pass for line plots — apply after `ax.plot(...)`."""
    apply_axis_style(ax)
    ax.grid(False)


def palette_list(*, include_accent: bool = True) -> list[str]:
    """Return the ordered full palette as a list of hex strings.

    Useful for matplotlib's default color cycle: `plt.rcParams['axes.prop_cycle']`.
    """
    out: list[str] = []
    if include_accent:
        out.extend([ACCENT_BLUE, ACCENT_BLUE_DARK])
    out.extend(PASTEL[k] for k in PASTEL_ORDER)
    out.extend([DARK_GRAY, MID_GRAY])
    return out


# ---------------------------------------------------------------------------
# Public re-export list
# ---------------------------------------------------------------------------

__all__ = [
    "ACCENT_BLUE", "ACCENT_BLUE_LIGHT", "ACCENT_BLUE_DARK",
    "ACCENT_YELLOW", "ACCENT_YELLOW_DARK",
    "DARK_GRAY", "MID_GRAY", "LIGHT_GRAY", "VERY_LIGHT_GRAY",
    "BLACK", "WHITE",
    "PASTEL", "PASTEL_ORDER", "FAMILY_COLORS",
    "get_part_color", "apply_axis_style",
    "style_barplot", "style_scatter", "style_lineplot",
    "palette_list",
]
