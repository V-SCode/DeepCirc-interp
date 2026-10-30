"""Rebuild every requested source panel, pad at native scale, then compose.

Unlike the historical preview composer, this entry point rejects missing
source or raster panels before producing a current setFinalvF preview.
Outputs are source panels and review previews, not Illustrator final artwork.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
VF = ROOT / "figures/setFinalvF"
sys.path.insert(0, str(ROOT))
from figures.figtools.layout import load_manifest
from figures.scripts.build_figure import main as compose

SCRIPTS = {
    10: [f"build_panel_{letter}.py" for letter in "abcde"],
    11: ["build_panel_d_stacked.py", "build_panel_e.py"],
}


def require_manifest_panels(path: Path) -> None:
    manifest = load_manifest(path)
    for panel in manifest.panels:
        vector = (path.parent / panel.file).resolve()
        preview = panel.resolve_preview_path(path.parent)
        if not vector.is_file() or preview is None or not preview.is_file():
            raise FileNotFoundError(f"Incomplete {manifest.figure_id} panel {panel.id}: {vector}, {preview}")
        if not vector.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"Invalid vector panel: {vector}")
        if not preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Invalid raster panel: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("10", "11", "all"), default="all")
    args = parser.parse_args()
    numbers = [10, 11] if args.figure == "all" else [int(args.figure)]
    for number in numbers:
        for filename in SCRIPTS[number]:
            runpy.run_path(str(VF / f"figS{number}/scripts" / filename), run_name="__main__")
    padding_path = ROOT / "figures/illustrator/prepare_vf_publication.py"
    spec = importlib.util.spec_from_file_location("native_panel_padding", padding_path)
    padding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(padding)
    padding.FIGURES = [entry for entry in padding.FIGURES
                       if int(entry["source_figure"].removeprefix("figS")) in numbers]
    padding.prepare()
    for number in numbers:
        manifest = VF / f"figS{number}/manifest.yaml"
        require_manifest_panels(manifest)
        if compose(["--manifest", str(manifest)]):
            raise RuntimeError(f"Could not compose {manifest}")


if __name__ == "__main__":
    main()
