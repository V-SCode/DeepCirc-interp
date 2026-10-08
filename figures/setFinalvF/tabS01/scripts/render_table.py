"""Render Table S1 as an editable PDF/SVG and a 600-dpi white RGB PNG.

The authoritative CSV is read unchanged. Layout and part colors are shared
with Table S2 and follow the delivered FigS13v2 and FigS15 figures.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.setFinalvF._shared.design_tables import render_design_table


def build_table() -> None:
    out_base = SCRIPT_PATH.parents[1] / "final" / "table_S1"
    with out_base.with_suffix(".csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    render_design_table(rows, out_base, number=1)


if __name__ == "__main__":
    build_table()
