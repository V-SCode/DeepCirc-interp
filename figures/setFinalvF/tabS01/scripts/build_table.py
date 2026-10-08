"""Supplementary Table S1 — Top 15 max-circuit designs (one per target function).

Derived from the top-right table of setFinal/figS13/panel_c. Same
selection logic: filter Pareto-front designs by paper MLP-gate floors
(circuit_score >= 2, growth >= 0.5), deduplicate to 1 design per target,
rank by circuit_log, keep top 15. Part names are reordered by graph
position (input-proximal → output-adjacent) so the same slot semantics
as figS13 hold row-to-row.

Output:
  final/table_S1.csv
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact helper functions from the setFinal panel script so the
# table stays bit-identical to what was rendered in the figure.
sys.path.insert(0, str(PKG_ROOT / "figures" / "setFinal" / "figS13" / "scripts"))
from build_panel_c import (  # noqa: E402
    _top15_maxcirc_per_target,
    _compute_slot_d2o,
    _reorder_parts_by_position,
)


def build_table() -> pd.DataFrame:
    maxc = _top15_maxcirc_per_target()

    graphs = json.loads((REPO_ROOT / "data" / "topology_g3"
                          / "topology_graphs.json").read_text())
    topo_by_id = {t["topology_id"]: t for t in graphs["topologies"]}

    rows = []
    for _, r in maxc.iterrows():
        parts = (ast.literal_eval(r["part_names"])
                 if isinstance(r["part_names"], str)
                 else list(r["part_names"]))
        slot_d2o = _compute_slot_d2o(topo_by_id[r["topology_id"]])
        parts_ordered = _reorder_parts_by_position(parts, slot_d2o)

        rows.append({
            "rank":          int(r["table_rank"]),
            "target":        r["target"],
            "size":          f"{int(len(parts))}-reg",
            "circuit_score": round(float(r["circuit_score"]), 2),
            "growth_score":  round(float(r["toxicity"]), 3),
            "part_names":    " → ".join(parts_ordered),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_table()
    out = SCRIPT_PATH.parents[1] / "final" / "table_S1.csv"
    df.to_csv(out, index=False)
    print(f"Wrote: {out.relative_to(REPO_ROOT)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
