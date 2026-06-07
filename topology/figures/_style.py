"""Locked style + palette + figure-save conventions for the G1 figure pipeline.

Imported by every fig0N_*.py script so palettes / fonts / sizes / save paths
stay consistent across the report. Reuse for G2 / G3 by passing --group at
the script level; this module is group-agnostic.

Design choices (post-discussion 2026-05-05):
- Colorblind-friendly target palette (Okabe-Ito-derived).
- Two-tone diverging for log_odds heatmaps (vermillion ↔ teal, white at 0).
- Single sans-serif throughout (matplotlib falls back gracefully if
  Helvetica/Arial isn't on the system).
- Save both PDF (vector) and PNG (300 dpi) per figure.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

# Targets — colorblind-safe, distinct hues. G1 uses Okabe-Ito 4-color subset;
# G2-new additions use the remaining Okabe-Ito hues + Tol-Vibrant; G3-new
# additions are pre-allocated so the palette doesn't shift when G3 fires.
# Source palettes:
#   - Okabe-Ito (https://jfly.uni-koeln.de/color/) — colorblind-safe categorical
#   - Tol Vibrant + Muted (https://personal.sron.nl/~pault/) — colorblind-safe
#
# Convention: each target gets ONE locked hue across G1/G2/G3. Adding a new
# target only requires a new entry; existing targets never have their colors
# reassigned. This keeps figure cross-referencing stable as the substrate
# scales.
TARGET_COLORS = {
    # G1 (4 targets)
    "0xEE": "#0072B2",   # Okabe-Ito blue       — easy 2-input OR (sym)
    "0x17": "#E69F00",   # Okabe-Ito orange     — asym 3-input (anchor)
    "0x2B": "#009E73",   # Okabe-Ito green      — asym 3-input (anchor)
    "0x6D": "#D55E00",   # Okabe-Ito vermillion — asym 3-input (hardest, 7-reg only)
    # G2-new (6 additional targets)
    "0x4D": "#56B4E9",   # Okabe-Ito sky blue   — NPN 0x17 variant (asym 3-input, 4th rep of class)
    "0xCC": "#CC79A7",   # Okabe-Ito red-purple — f=B (1-input degenerate, NPN 0x0F)
    "0xE8": "#F0E442",   # Okabe-Ito yellow     — majority(A,B,C) (sym, iconic)
    "0x33": "#117733",   # Tol Vibrant green    — partial-symmetric XOR-like
    "0x66": "#882255",   # Tol Vibrant wine     — XOR(B,C), input A degenerate (NPN 0x3C)
    "0x3C": "#44AA99",   # Tol Vibrant teal     — partial-symmetric pair
    # G3-new (10 additional targets — locked 2026-05-09 after the G3 P1 in-scope
    # gate; original 0x18 (NPN 0x18) + 0xB4 (NPN 0x1E) failed structural-floor
    # at 4-7 reg and were swapped to 0x71 + 0x8E from NPN 0x17 family. The
    # 0x18/0xB4 palette slots transfer to the swap-ins so figure colors stay
    # consistent across the v2.0 substrate evolution.)
    "0x2C": "#332288",   # Tol Vibrant indigo
    "0x03": "#999933",   # Tol Vibrant olive
    "0x0F": "#AA4499",   # Tol Vibrant magenta
    "0x06": "#6699CC",   # Tol Muted soft blue
    "0x07": "#CC6677",   # Tol Vibrant rose
    "0x71": "#88CCEE",   # Tol Muted light cyan       (slot from dropped 0x18)
    "0x8E": "#DDCC77",   # Tol Vibrant sand           (slot from dropped 0xB4)
    "0x77": "#AA7744",   # Tol Muted brown
    "0x47": "#661100",   # Tol Muted dark red
    "0x1B": "#777777",   # neutral medium gray
    # Special
    "baseline": "#BBBBBB",  # light gray — baseline-derived (deferred under v2.0+)
}

# NPN-class membership per target (canonical NPN class label per target_function_selection.md).
# Used to group targets by NPN equivalence in figures. Stable across G1/G2/G3.
TARGET_NPN_CLASS = {
    "0xEE": "0x03", "0x17": "0x17", "0x2B": "0x17", "0x6D": "0x16",
    "0x4D": "0x17", "0xCC": "0x0F", "0xE8": "0x17", "0x33": "0x0F",
    "0x66": "0x3C", "0x3C": "0x3C",
    "0x2C": "0x19", "0x03": "0x03", "0x0F": "0x0F", "0x06": "0x06",
    "0x07": "0x07", "0x71": "0x17", "0x8E": "0x17", "0x77": "0x03",
    "0x47": "0x1B", "0x1B": "0x1B",
}

# Display order for NPN classes (by population size in G3, then by canonical hex).
NPN_CLASS_ORDER = ["0x17", "0x03", "0x0F", "0x1B", "0x3C", "0x16", "0x06", "0x07", "0x19"]

# Distinct color per NPN class — colorblind-safe, derived from Tol Bright/Vibrant.
NPN_CLASS_COLORS = {
    "0x17": "#4477AA",   # Tol Bright blue        — NOR-friendly asym 3-input (6 targets)
    "0x03": "#228833",   # Tol Bright green       — 2-input symmetric (3 targets)
    "0x0F": "#AA3377",   # Tol Bright purple      — 1-input degenerate (3 targets)
    "0x1B": "#CCBB44",   # Tol Bright yellow      — HW=4 asym (2 targets)
    "0x3C": "#66CCEE",   # Tol Bright cyan        — 2-input partial (2 targets)
    "0x16": "#EE6677",   # Tol Bright rose        — paper anchor 0x6D (1 target)
    "0x06": "#BBBBBB",   # neutral gray           — XOR-fragment HW=2 (1 target)
    "0x07": "#EE7733",   # Tol Vibrant orange     — 3-minterm HW=3 (1 target)
    "0x19": "#882255",   # Tol Vibrant wine       — smoke-test 0x2C (1 target)
}


def parse_source(source_val) -> str:
    """Normalize a source field to a single canonical hex.

    G3 has multi-source rows (e.g., ["0x47","0x1B"] for shared NPN-equiv
    topologies). For figures we collapse to the first target listed —
    same convention as the rule book's per-target attribution.

    Handles both JSON-encoded lists (double quotes) AND Python-repr
    lists (single quotes) when the source is stored as a CSV string.
    """
    if isinstance(source_val, list):
        return str(source_val[0]) if source_val else "?"
    s = str(source_val)
    if s.startswith("[") and s.endswith("]"):
        import json as _json
        import ast as _ast
        # Try JSON first, then Python-literal fallback for ['x', 'y']-style.
        for parser in (_json.loads, _ast.literal_eval):
            try:
                v = parser(s)
                if isinstance(v, (list, tuple)) and v:
                    return str(v[0])
            except (ValueError, SyntaxError):
                continue
    if s.startswith("baseline:"):
        return s.split(":", 1)[1].split("/", 1)[0]
    return s


def npn_class_of(source_val) -> str:
    """Return the NPN class for a (possibly multi-)source value."""
    return TARGET_NPN_CLASS.get(parse_source(source_val), "?")


# Part-class colors — used for highlighting in beeswarms / bar charts.
PART_CLASS_COLORS = {
    "toxic_extreme":  "#A50026",  # IcaRA/I1 (deep red)
    "toxic_tier2":    "#E66101",  # QacR/Q1, HlyIIR/H1 (orange-red)
    "protective":     "#1A9850",  # QacR/Q2, PhlF/P1 (green)
    "circuit_3a":     "#2166AC",  # AmeR/F1 (blue)
    "circuit_3b":     "#5AAE61",  # PhlF/P2-P3 (light green)
    "neutral":        "#BDBDBD",  # most parts (gray)
}

# Motif edge-pattern colors (for v1.3 figure 5).
MOTIF_COLORS = {
    "good":      "#1A9850",   # green — internal-only / protective motifs
    "bad":       "#D73027",   # red   — sensor-driven / OUT-attached motifs
    "neutral":   "#BDBDBD",
    "saturated": "#F4A582",   # peach — saturation-flagged cells
}

# Diverging colormap for log_odds heatmaps (red ↔ white ↔ blue).
LOGODDS_CMAP = "RdBu_r"

# QC tier colors. A2 was originally pale yellow (#FEE08B); too light for
# print/contrast, replaced with vivid amber.
QC_TIER_COLORS = {
    "A1": "#1A9850",   # green        — paper-grade fit
    "A2": "#F57C00",   # vivid amber  — borderline
    "A3": "#D73027",   # red          — hard-to-fit (excluded)
    "A0": "#7B7B7B",   # gray         — marker (no MLP)
}


# ---------------------------------------------------------------------------
# Typography & figure sizes
# ---------------------------------------------------------------------------

FONT_FAMILY = ["Helvetica", "Arial", "DejaVu Sans"]
TITLE_SIZE  = 11
LABEL_SIZE  = 10
TICK_SIZE   = 8
PANEL_LABEL_SIZE = 12
LEGEND_SIZE = 8
ANNOTATION_SIZE = 8

FIG_SIZES = {
    "1col":  (3.5, 2.7),   # single panel, narrow
    "1col_tall":  (3.5, 4.5),
    "2col":  (7.2, 5.4),   # 2x2 grid, paper full width
    "3row":  (7.2, 9.0),   # 3-row layout (Fig 1)
    "wide":  (7.2, 3.5),   # wide single row
}

PANEL_LABELS = list("ABCDEFGH")


def setup_matplotlib() -> None:
    """Apply the locked rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   FONT_FAMILY,
        "font.size":         LABEL_SIZE,
        "axes.titlesize":    TITLE_SIZE,
        "axes.labelsize":    LABEL_SIZE,
        "xtick.labelsize":   TICK_SIZE,
        "ytick.labelsize":   TICK_SIZE,
        "legend.fontsize":   LEGEND_SIZE,
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "lines.linewidth":   1.4,
        "patch.linewidth":   0.6,
        "pdf.fonttype":      42,   # editable text in PDF
        "ps.fonttype":       42,
    })


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def output_dir(group: str = "G1") -> Path:
    """Return the per-group output directory, creating it if needed."""
    p = Path(__file__).parent / "output" / group.lower()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig, name: str, group: str = "G1", *, transparent: bool = False) -> tuple[Path, Path]:
    """Write the figure as both PDF (vector) and PNG (300 dpi)."""
    outdir = output_dir(group)
    pdf_path = outdir / f"{name}.pdf"
    png_path = outdir / f"{name}.png"
    fig.savefig(pdf_path, transparent=transparent)
    fig.savefig(png_path, transparent=transparent)
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Panel-label utility
# ---------------------------------------------------------------------------

def panel_label(ax, letter: str, *, x: float = -0.12, y: float = 1.05, weight: str = "bold") -> None:
    """Add a panel letter (A, B, C, ...) at the top-left of an axis."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
            fontweight=weight, va="bottom", ha="left")


# Locked convention: figure title sits in the top-left corner of every figure.
# Use figure_title() instead of fig.suptitle() so this stays consistent.
TITLE_X = 0.02
TITLE_Y = 0.985


def figure_title(fig, text: str, *, fontsize: int | None = None) -> None:
    """Place the figure title at top-left of the figure boundary.

    All figures in the G1/G2/G3 pipeline call this instead of fig.suptitle()
    so the convention stays consistent. Position is locked at x=0.02, y=0.985.
    """
    fig.text(TITLE_X, TITLE_Y, text,
             fontsize=fontsize or (TITLE_SIZE + 3),
             fontweight="bold",
             ha="left", va="top")
