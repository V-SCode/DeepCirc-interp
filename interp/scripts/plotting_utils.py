"""Shared plotting utilities for DeepCircMI figures.

Two concerns:
  1. Consistent colors and style across figures so cross-figure reading works.
  2. Reproducibility: every figure emits a `.data.json` sidecar with the
     exact numbers plotted, so a paper-quality redraw doesn't need to
     re-run the upstream analysis.

Exposed:
  FAMILY_PALETTE        — {family_name: hex} consistent across all figures.
  SCORE_COLORS          — circuit / growth / ON / OFF / toxic / favorable.
  apply_exploratory_style()  — screen-reading defaults used during analysis.
  apply_paper_style()        — journal-ready locked fonts/sizes (WIP — tune
                                per journal once target is chosen).
  save_figure_with_data(fig, path_prefix, data, *, script_name=None, ...)
                             — writes PNG (+ optional PDF) and `.data.json`
                                sidecar with provenance (git HEAD, script,
                                timestamp) + the data dict as provided.

Sidecar schema:
  {
    "provenance": {"script": str|None, "git_head": str|None,
                   "created_utc": ISO-8601},
    "data": <whatever caller passed>
  }

Callers decide what `data` contains — should be enough to re-plot without
rerunning analysis. Parse-friendly (lists, floats, dicts) — no numpy arrays.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt


# Sidecar JSONs live in a sibling directory to `figures/` so the figure
# folder stays clean (images + PDFs only). Set via env var for tests.
_REPO_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = _REPO_ROOT / "figures"
FIGS_JSON_DIR = _REPO_ROOT / "figures_json"


# ---------------------------------------------------------------------------
# Consistent palettes
# ---------------------------------------------------------------------------

FAMILY_PALETTE: dict[str, str] = {
    "AmeR":   "#7B1FA2",
    "AmtR":   "#C2185B",
    "BetI":   "#FFA000",
    "BM3R1":  "#388E3C",
    "HlyIIR": "#0097A7",
    "IcaRA":  "#D32F2F",
    "LitR":   "#5D4037",
    "LmrA":   "#455A64",
    "PhlF":   "#1976D2",
    "PsrA":   "#00796B",
    "QacR":   "#F57C00",
    "SrpR":   "#512DA8",
}

SCORE_COLORS: dict[str, str] = {
    "circuit":   "#3478f6",  # blue
    "growth":    "#888888",  # gray
    "on":        "#1f9d55",  # green (truth-table ON)
    "off":       "#c81d25",  # red (truth-table OFF)
    "toxic":     "#b2182b",  # red for +α burden / growth-toxic
    "favorable": "#2166ac",  # blue for −α burden / growth-favorable
}


# ---------------------------------------------------------------------------
# Style registers
# ---------------------------------------------------------------------------

def apply_exploratory_style() -> None:
    """Screen-reading defaults used during analysis. Matches what most
    existing figures already use — so new figures render consistently."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })


def apply_paper_style() -> None:
    """Journal-ready locked fonts/sizes. Tune to specific journal conventions
    once the target venue is chosen; these are Nature-ish defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "figure.titleweight": "normal",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 3,
        "xtick.major.width": 0.7,
        "ytick.major.size": 3,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,   # TrueType — many journals require this
        "ps.fonttype": 42,
    })


# ---------------------------------------------------------------------------
# Figure + data sidecar writer
# ---------------------------------------------------------------------------

def _git_head(cwd: Optional[Path] = None) -> Optional[str]:
    """Short git SHA if we're inside a repo; otherwise None."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def save_figure_with_data(
    fig,
    path_prefix: Path | str,
    data: Optional[Mapping[str, Any]] = None,
    *,
    source_files: Optional[list[tuple[str, str] | str]] = None,
    how_to_replot: Optional[str] = None,
    script_name: Optional[str] = None,
    dpi: int = 150,
    write_pdf: bool = True,
    facecolor: str = "white",
    json_dir: Optional[Path | str] = None,
) -> Path:
    """Save figure + PDF + data sidecar.

    Writes:
      {figures/}{basename}.png
      {figures/}{basename}.pdf         (if write_pdf=True)
      {figures_json/}{basename}.data.json

    Two modes for the sidecar (mutually compatible — you can pass both):
      1. `data=` a dict of JSON-serializable values — the plot's numbers
         inlined for redraws without upstream deps.
      2. `source_files=` a list of paths (strings) or (path, description)
         tuples pointing at upstream JSON/parquet files that hold the
         same numbers — avoids duplicating large arrays when the plot
         simply visualizes an existing `processed/` file.
    Either or both is fine.

    `path_prefix` should live under `figures/`; the sidecar JSON is routed
    to the sibling `figures_json/` directory by default so the figure
    folder stays clean.
    """
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    png_path = path_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    if write_pdf:
        pdf_path = path_prefix.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=facecolor)

    json_root = Path(json_dir) if json_dir is not None else FIGS_JSON_DIR
    json_root.mkdir(parents=True, exist_ok=True)
    json_path = json_root / f"{path_prefix.stem}.data.json"

    sidecar: dict[str, Any] = {
        "provenance": {
            "script": script_name,
            "git_head": _git_head(path_prefix.parent),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "figure_file": str(png_path.relative_to(_REPO_ROOT)),
        },
    }
    if data is not None:
        sidecar["data"] = dict(data)
    if source_files is not None:
        sidecar["source_files"] = [
            {"path": p, "description": ""} if isinstance(p, str)
            else {"path": p[0], "description": p[1]}
            for p in source_files
        ]
    if how_to_replot is not None:
        sidecar["how_to_replot"] = how_to_replot

    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)

    return png_path


__all__ = [
    "FAMILY_PALETTE", "SCORE_COLORS",
    "FIGS_DIR", "FIGS_JSON_DIR",
    "apply_exploratory_style", "apply_paper_style",
    "save_figure_with_data",
]
