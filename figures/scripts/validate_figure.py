"""CLI wrapper for `figtools.validate`.

Usage:
    python figures/scripts/validate_figure.py \\
        --manifest figures/demo/manifest.yaml

Exit codes:
  0 : no findings, or only INFO
  1 : at least one WARN, no ERRORs
  2 : at least one ERROR (or runtime failure)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.figtools.layout import load_manifest  # noqa: E402
from figures.figtools.validate import (  # noqa: E402
    run_all, render_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a figure manifest.")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Path to manifest.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest.exists():
        print(f"[ERROR] manifest not found: {args.manifest}")
        return 2
    manifest = load_manifest(args.manifest)
    findings = run_all(manifest)
    print(render_report(manifest, findings))

    has_err = any(f.severity == "ERROR" for f in findings)
    has_warn = any(f.severity == "WARN" for f in findings)
    if has_err:
        return 2
    if has_warn:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
