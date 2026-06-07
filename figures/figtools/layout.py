"""Manifest-driven figure layout.

A figure manifest is a YAML file describing:
- the canvas size in mm
- a list of panels, each with file path + position (mm) + size (mm)
- optional panel-label text + offset (mm) for each panel

The pipeline composes panels into a single preview PDF + PNG by loading
each panel's raster preview (PNG) and placing it on a millimeter-precise
matplotlib canvas. The Illustrator stage uses the same manifest, but
links the vector PDFs/SVGs from `panels/vector/` instead.

Manifest schema (YAML):
    figure_id: figX
    canvas:
      width_mm: 181.8
      height_mm: 109.7
    panels:
      - id: a
        file: panels/vector/panel_a.pdf
        preview_file: panels/raster/panel_a.png   # optional override
        x_mm: 0
        y_mm: 0
        w_mm: 60
        h_mm: 35
        label:
          text: a
          x_offset_mm: -3
          y_offset_mm: -2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .export import mm_to_in
from ..styles.colors import DARK_GRAY
from ..styles.typography import PANEL_LABEL_SIZE, DEFAULT_FONT_FAMILY


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PanelLabelSpec:
    text: str
    x_offset_mm: float = -3.0
    y_offset_mm: float = -2.0
    size_pt: float = PANEL_LABEL_SIZE
    weight: str = "bold"


@dataclass
class PanelSpec:
    id: str
    file: Path                    # canonical vector asset (PDF or SVG)
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    label: PanelLabelSpec | None = None
    preview_file: Path | None = None  # raster preview (PNG); falls back to `file`

    @property
    def x1_mm(self) -> float:
        return self.x_mm + self.w_mm

    @property
    def y1_mm(self) -> float:
        return self.y_mm + self.h_mm

    def resolve_preview_path(self, manifest_dir: Path) -> Path | None:
        """Return the absolute preview PNG path if it exists."""
        candidate = self.preview_file
        if candidate is None:
            # convention: panels/vector/panel_a.pdf -> panels/raster/panel_a.png
            v = (manifest_dir / self.file).resolve()
            candidate = v.parent.parent / "raster" / (v.stem + ".png")
        else:
            candidate = (manifest_dir / candidate).resolve()
        return candidate if candidate.exists() else None


@dataclass
class Manifest:
    figure_id: str
    canvas_width_mm: float
    canvas_height_mm: float
    panels: list[PanelSpec] = field(default_factory=list)
    source_path: Path = Path()

    @property
    def manifest_dir(self) -> Path:
        return self.source_path.parent

    def linked_asset_sidecar(self) -> list[dict[str, Any]]:
        """Materialize a JSON-serializable description of all linked assets.

        Used by `scripts/build_figure.py` to drop a sidecar next to the
        preview so Illustrator can place the same files.
        """
        out = []
        for p in self.panels:
            entry: dict[str, Any] = {
                "id": p.id,
                "file": str(p.file),
                "x_mm": p.x_mm, "y_mm": p.y_mm,
                "w_mm": p.w_mm, "h_mm": p.h_mm,
            }
            if p.label is not None:
                entry["label"] = {
                    "text": p.label.text,
                    "x_offset_mm": p.label.x_offset_mm,
                    "y_offset_mm": p.label.y_offset_mm,
                    "size_pt": p.label.size_pt,
                    "weight": p.label.weight,
                }
            if p.preview_file is not None:
                entry["preview_file"] = str(p.preview_file)
            out.append(entry)
        return out


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> Manifest:
    """Parse a YAML manifest into a Manifest dataclass."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    canvas = raw.get("canvas", {})
    panels: list[PanelSpec] = []
    for entry in raw.get("panels", []):
        label_data = entry.get("label")
        label: PanelLabelSpec | None = None
        if label_data:
            label = PanelLabelSpec(
                text=str(label_data.get("text", entry.get("id", ""))),
                x_offset_mm=float(label_data.get("x_offset_mm", -3.0)),
                y_offset_mm=float(label_data.get("y_offset_mm", -2.0)),
                size_pt=float(label_data.get("size_pt", PANEL_LABEL_SIZE)),
                weight=str(label_data.get("weight", "bold")),
            )
        panels.append(PanelSpec(
            id=str(entry["id"]),
            file=Path(entry["file"]),
            x_mm=float(entry["x_mm"]),
            y_mm=float(entry["y_mm"]),
            w_mm=float(entry["w_mm"]),
            h_mm=float(entry["h_mm"]),
            label=label,
            preview_file=Path(entry["preview_file"]) if entry.get("preview_file") else None,
        ))

    return Manifest(
        figure_id=str(raw.get("figure_id", path.parent.name)),
        canvas_width_mm=float(canvas.get("width_mm", 181.8)),
        canvas_height_mm=float(canvas.get("height_mm", 109.7)),
        panels=panels,
        source_path=path.resolve(),
    )


def save_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Serialize a Manifest back to YAML."""
    path = Path(path)
    payload: dict[str, Any] = {
        "figure_id": manifest.figure_id,
        "canvas": {
            "width_mm": manifest.canvas_width_mm,
            "height_mm": manifest.canvas_height_mm,
        },
        "panels": [],
    }
    for p in manifest.panels:
        entry: dict[str, Any] = {
            "id": p.id,
            "file": str(p.file),
            "x_mm": p.x_mm, "y_mm": p.y_mm,
            "w_mm": p.w_mm, "h_mm": p.h_mm,
        }
        if p.preview_file is not None:
            entry["preview_file"] = str(p.preview_file)
        if p.label is not None:
            entry["label"] = {
                "text": p.label.text,
                "x_offset_mm": p.label.x_offset_mm,
                "y_offset_mm": p.label.y_offset_mm,
            }
        payload["panels"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# Preview assembly
# ---------------------------------------------------------------------------

def assemble_preview(
    manifest: Manifest,
    *,
    output_pdf: Path | str | None = None,
    output_png: Path | str | None = None,
    background: str = "white",
    show_panel_outlines: bool = False,
) -> dict[str, Path]:
    """Compose a single preview figure from the panel raster previews.

    Each panel is loaded from its `raster/<id>.png` preview (Illustrator
    will later relink to the vector PDF). Position + size are mm-precise.

    Returns the dict of written paths {format: Path}.
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(
        figsize=(mm_to_in(manifest.canvas_width_mm),
                 mm_to_in(manifest.canvas_height_mm)),
        dpi=300,
    )
    fig.patch.set_facecolor(background)

    canvas_w = manifest.canvas_width_mm
    canvas_h = manifest.canvas_height_mm

    for p in manifest.panels:
        # axes in fractional fig coords; matplotlib uses (left, bottom, w, h)
        # with origin at bottom-left, while manifest uses top-left origin.
        left   = p.x_mm / canvas_w
        bottom = 1.0 - (p.y_mm + p.h_mm) / canvas_h
        width  = p.w_mm / canvas_w
        height = p.h_mm / canvas_h
        ax = fig.add_axes((left, bottom, width, height))
        ax.set_xticks([])
        ax.set_yticks([])

        if show_panel_outlines:
            for spine in ax.spines.values():
                spine.set_color(DARK_GRAY)
                spine.set_linewidth(0.3)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

        preview_path = p.resolve_preview_path(manifest.manifest_dir)
        if preview_path is not None:
            img = mpimg.imread(str(preview_path))
            ax.imshow(img, aspect="auto", interpolation="bilinear")
        else:
            ax.text(0.5, 0.5,
                    f"[missing preview]\n{p.id}\n{p.file}",
                    transform=ax.transAxes,
                    fontsize=5, ha="center", va="center",
                    color="#cc4444",
                    family=DEFAULT_FONT_FAMILY[0])

        if p.label is not None:
            label_x_frac = (p.x_mm + p.label.x_offset_mm) / canvas_w
            label_y_frac = 1.0 - (p.y_mm + p.label.y_offset_mm) / canvas_h
            fig.text(
                label_x_frac, label_y_frac,
                str(p.label.text).lower(),
                fontsize=p.label.size_pt,
                fontweight=p.label.weight,
                ha="left", va="top",
                color=DARK_GRAY,
                family=DEFAULT_FONT_FAMILY[0],
            )

    written: dict[str, Path] = {}
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, format="pdf", dpi=600, transparent=False,
                     facecolor=background)
        written["pdf"] = output_pdf
    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, format="png", dpi=300, transparent=False,
                     facecolor=background)
        # Convert to RGB to drop the alpha channel — Word renders a visible
        # picture-frame border around any RGBA PNG with transparent edges,
        # even when the alpha is fully opaque.
        from PIL import Image
        Image.open(output_png).convert("RGB").save(output_png, "PNG")
        written["png"] = output_png

    plt.close(fig)
    return written


__all__ = [
    "PanelLabelSpec", "PanelSpec", "Manifest",
    "load_manifest", "save_manifest", "assemble_preview",
]
