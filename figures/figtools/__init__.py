"""DeepCirc figure-generation pipeline — code library.

Submodules:
- export      : save_panel(), use_style(), mm <-> inch conversions
- plots       : reusable matplotlib plotting primitives (bar, scatter, ...)
- schematics  : SVG-based circuit / workflow components
- layout      : YAML manifest -> figure-canvas assembly
- validate    : sanity checks for manifest + asset files

Convention: import the high-level helpers you need at the top of your
panel-building script. The pipeline never mutates raw data; it only
formats it.
"""
from __future__ import annotations

# Force a non-interactive backend for headless figure generation. macOS's
# `macosx` backend tries to allocate a window even from a CLI script,
# which crashes when no display is attached. Override before any matplotlib
# import happens. Users can override by setting MPLBACKEND externally.
import os as _os
_os.environ.setdefault("MPLBACKEND", "Agg")

# Re-export the most commonly used helpers so user scripts can do
#     from figtools import save_panel, use_style, MM_PER_IN
from .export import (  # noqa: F401
    save_panel, use_style,
    mm_to_in, in_to_mm, mm_to_pt, figsize_mm,
    MM_PER_IN, PT_PER_IN,
)

__all__ = [
    "save_panel", "use_style",
    "mm_to_in", "in_to_mm", "mm_to_pt", "figsize_mm",
    "MM_PER_IN", "PT_PER_IN",
]
