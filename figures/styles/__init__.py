"""DeepCirc paper figure style package.

Submodules:
- colors      : palette constants + axis-style helpers
- typography  : font-size constants + panel-label helpers

The matplotlib stylesheet lives at `figures/styles/deepcirc.mplstyle`
and is loaded via `figtools.export.use_style()` or directly with
`plt.style.use("figures/styles/deepcirc.mplstyle")`.
"""
from . import colors, typography  # noqa: F401

__all__ = ["colors", "typography"]
