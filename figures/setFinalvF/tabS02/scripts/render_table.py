"""Render Table S2 and its part-frequency summary as PDF, SVG and PNG.

Both sections are read directly from the authoritative CSV. The shared
renderer preserves values, ordering and the paper's categorical part colors.
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


def _load_main_and_footer():
    csv_path = SCRIPT_PATH.parents[1] / "final" / "table_S2.csv"
    with csv_path.open(newline="") as source:
        rows = list(csv.reader(source))
    separator = next(i for i, row in enumerate(rows) if not any(row))
    main = [dict(zip(rows[0], row)) for row in rows[1:separator]]
    footer_header = rows[separator + 2]
    footer = [dict(zip(footer_header, row)) for row in rows[separator + 3:] if row]
    return main, footer


def build_table() -> None:
    main, footer = _load_main_and_footer()
    out_base = SCRIPT_PATH.parents[1] / "final" / "table_S2"
    render_design_table(main, out_base, number=2, footer=footer)


if __name__ == "__main__":
    build_table()
