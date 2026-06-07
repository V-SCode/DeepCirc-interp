"""Assemble a figure preview from a manifest.

Reads a YAML manifest, places each panel's raster preview on a millimeter-
precise canvas, and writes:
  <figure_dir>/final/<figure_id>.pdf
  <figure_dir>/final/<figure_id>.png
  <figure_dir>/final/<figure_id>.linked_assets.json   (Illustrator sidecar)

The PDF is intentionally a *preview*. Vector quality is preserved when
panels are individually placed/linked in Illustrator using the same
manifest — see illustrator/place_linked_assets_from_manifest.jsx.

Usage:
    python figures/scripts/build_figure.py \\
        --manifest figures/demo/manifest.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools.export import use_style  # noqa: E402
from figures.figtools.layout import (  # noqa: E402
    load_manifest, assemble_preview,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble a figure preview from a YAML manifest.")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Path to the figure's manifest.yaml")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Override output directory (default: <manifest_dir>/final/)")
    p.add_argument("--show-panel-outlines", action="store_true",
                   help="Draw thin outlines around each panel placement (debug)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    use_style()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}")
        return 2

    manifest = load_manifest(manifest_path)
    out_dir = (args.outdir or manifest.manifest_dir / "final").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{manifest.figure_id}.pdf"
    png_path = out_dir / f"{manifest.figure_id}.png"
    sidecar  = out_dir / f"{manifest.figure_id}.linked_assets.json"

    print(f"assembling `{manifest.figure_id}` "
          f"({manifest.canvas_width_mm:.1f} mm x {manifest.canvas_height_mm:.1f} mm, "
          f"{len(manifest.panels)} panels)")
    written = assemble_preview(
        manifest,
        output_pdf=pdf_path,
        output_png=png_path,
        show_panel_outlines=args.show_panel_outlines,
    )

    # Sidecar JSON listing all linked assets for the Illustrator step.
    sidecar_payload = {
        "figure_id": manifest.figure_id,
        "canvas_width_mm": manifest.canvas_width_mm,
        "canvas_height_mm": manifest.canvas_height_mm,
        "panels": manifest.linked_asset_sidecar(),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
    }
    with sidecar.open("w") as f:
        json.dump(sidecar_payload, f, indent=2)

    for fmt, path in written.items():
        print(f"  wrote {path.relative_to(REPO_ROOT)} ({fmt})")
    print(f"  wrote {sidecar.relative_to(REPO_ROOT)} (linked-asset sidecar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
