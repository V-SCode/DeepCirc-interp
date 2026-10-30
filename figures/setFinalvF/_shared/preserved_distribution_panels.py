"""Rebuild unchanged distribution panels using their frozen source scripts.

The temporary output directory prevents a current-set build from rewriting
the frozen setFinal figure outputs. Only the panel letter changes here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import tempfile


def build_distribution_panel(old_letter: str, new_letter: str) -> None:
    vf = Path(__file__).resolve().parents[1]
    old_script = vf.parent / "setFinal/figS10/scripts" / f"build_panel_{old_letter}.py"
    spec = importlib.util.spec_from_file_location(f"preserved_distribution_{old_letter}", old_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="deepcirc-distribution-") as temporary:
        panel_root = Path(temporary) / "panels"
        module.PANELS_VEC = panel_root / "vector"
        # DATA_PATH was resolved at import; this root only controls the
        # builder's relative output-path logging during this temporary build.
        module.REPO_ROOT = Path(temporary)
        module.build_panel()
        for suffix in ("pdf", "svg", "png"):
            folder = "raster" if suffix == "png" else "vector"
            source = panel_root / folder / f"panel_{old_letter}.{suffix}"
            destination = vf / "figS10/panels" / folder / f"panel_{new_letter}.{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            print(f"Preserved distribution panel: {destination}")
