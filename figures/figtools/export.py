"""Vector-first panel export helpers.

Every panel in this pipeline is exported as:
- PDF  (vector, for Illustrator placement)
- SVG  (vector, for inspection / SVG-based composition)
- PNG  (raster, for previews / contact sheets / smoke tests)

Use `save_panel(fig, "panels/vector/panel_a")` and the helper handles the
rest. The folder structure is convention:
    panels/vector/<name>.pdf
    panels/vector/<name>.svg
    panels/raster/<name>.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Constants ----------------------------------------------------------------

MM_PER_IN: float = 25.4
PT_PER_IN: float = 72.0

# Path of the canonical mplstyle, resolved at import time.
_STYLE_PATH = Path(__file__).resolve().parent.parent / "styles" / "deepcirc.mplstyle"


def mm_to_in(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm / MM_PER_IN


def in_to_mm(inches: float) -> float:
    """Convert inches to millimeters."""
    return inches * MM_PER_IN


def mm_to_pt(mm: float) -> float:
    """Convert millimeters to PostScript points."""
    return (mm / MM_PER_IN) * PT_PER_IN


def use_style() -> None:
    """Apply the locked DeepCirc paper style to matplotlib.

    Call this once at the top of any script that produces panels. It is
    idempotent — calling more than once is harmless.
    """
    import matplotlib.pyplot as plt

    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))
    else:
        # The mplstyle file should always exist alongside this module.
        # If it's missing we want a loud error so the user notices.
        raise FileNotFoundError(
            f"DeepCirc mplstyle missing at {_STYLE_PATH}. "
            "Re-run the pipeline setup or restore styles/deepcirc.mplstyle."
        )


def save_panel(
    fig,
    out_base: str | Path,
    *,
    vector: bool = True,
    raster: bool = True,
    dpi: int = 600,
    transparent: bool = True,
    close: bool = True,
    formats: Iterable[str] | None = None,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
) -> dict[str, Path]:
    """Export a Matplotlib Figure as PDF + SVG + PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_base : str | Path
        Base path WITHOUT extension. The function appends `.pdf`, `.svg`,
        `.png` as needed. Parent directories are created automatically.
    vector : bool
        If True (default), write `.pdf` and `.svg`.
    raster : bool
        If True (default), write `.png` at `dpi`.
    dpi : int
        Raster DPI. 600 is the default for the paper pipeline.
    transparent : bool
        Save with transparent backgrounds so panels composite cleanly.
    close : bool
        Close the figure after saving (default True — prevents the
        backend from accumulating open figures during batch builds).
    formats : iterable of str, optional
        Explicit format list, e.g. ["pdf", "png"]. Overrides vector / raster.

    Returns
    -------
    dict
        Mapping {format: Path} of files actually written.
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    if formats is None:
        chosen: list[str] = []
        if vector:
            chosen.extend(["pdf", "svg"])
        if raster:
            chosen.append("png")
    else:
        chosen = list(formats)

    # mplstyle pins rcParams["savefig.bbox"] = "tight" globally, so a
    # bare bbox_inches=None on savefig still uses tight. Use an
    # rc_context to force the requested behaviour when the caller
    # explicitly opts out of tight cropping (bbox_inches=None).
    import matplotlib as _mpl
    if bbox_inches is None:
        ctx = _mpl.rc_context({"savefig.bbox": None,
                                "savefig.pad_inches": pad_inches})
    else:
        # nullcontext-equivalent
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        for ext in chosen:
            # PNG goes to a sibling raster/ dir if the caller put us under panels/vector/
            path = _resolve_output_path(out_base, ext)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = dict(transparent=transparent, pad_inches=pad_inches)
            if bbox_inches is not None:
                save_kwargs["bbox_inches"] = bbox_inches
            if ext == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(path, format=ext, **save_kwargs)
            written[ext] = path

    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return written


def _resolve_output_path(out_base: Path, ext: str) -> Path:
    """Route .png to raster/, leave vectors in vector/ (if applicable).

    If `out_base` lives under a `vector/` directory and the format is PNG,
    swap to a sibling `raster/` directory to keep the panel folder tidy.
    Otherwise just append the extension to `out_base`.
    """
    if ext == "png" and out_base.parent.name == "vector":
        target_dir = out_base.parent.parent / "raster"
        return target_dir / (out_base.stem + "." + ext)
    return out_base.with_suffix("." + ext)


def figsize_mm(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Return a Matplotlib figsize (in inches) for a panel sized in mm."""
    return (mm_to_in(width_mm), mm_to_in(height_mm))


__all__ = [
    "MM_PER_IN", "PT_PER_IN",
    "mm_to_in", "in_to_mm", "mm_to_pt",
    "use_style", "save_panel", "figsize_mm",
]
