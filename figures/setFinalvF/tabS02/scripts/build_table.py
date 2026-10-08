"""Supplementary Table S2 — Top 15 Pareto-knee designs (one per target function),
ranked by combined score.

Derived from the bottom-right table of setFinal/figS13/panel_c. Selection
logic: filter each topology's Pareto knee at paper MLP-gate floors
(knee_A_circuit_log >= log2(2), knee_A_toxicity >= 0.5), recompute the
combined normalized (circuit + growth) score across all surviving knees,
deduplicate to 1 knee per target, rank by combined score, keep top 15.
Part names are reordered by graph position (input-proximal →
output-adjacent).

A footer section reports the Top-5 most frequent parts across the union
of the Table S1 (top-15 max-circuit) and Table S2 (top-15 Pareto-knee)
designs — the same list the source figS13 highlighted at the bottom of
the right panel. This is the load-bearing headline: these five parts
appear across the best-performing circuits regardless of whether they
are ranked by pure circuit strength or by joint circuit-and-growth
Pareto trade-off.

Output:
  final/table_S2.csv
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(PKG_ROOT / "figures" / "setFinal" / "figS13" / "scripts"))
from build_panel_c import (  # noqa: E402
    _top15_knee_per_target,
    _top15_maxcirc_per_target,
    _compute_slot_d2o,
    _reorder_parts_by_position,
)


def _reorder(topo_by_id: dict, r: pd.Series) -> list[str]:
    parts = (ast.literal_eval(r["part_names"])
             if isinstance(r["part_names"], str)
             else list(r["part_names"]))
    slot_d2o = _compute_slot_d2o(topo_by_id[r["topology_id"]])
    return _reorder_parts_by_position(parts, slot_d2o)


def build_table() -> pd.DataFrame:
    knee = _top15_knee_per_target()

    graphs = json.loads((REPO_ROOT / "data" / "topology_g3"
                          / "topology_graphs.json").read_text())
    topo_by_id = {t["topology_id"]: t for t in graphs["topologies"]}

    rows = []
    for _, r in knee.iterrows():
        parts_ordered = _reorder(topo_by_id, r)

        rows.append({
            "rank":           int(r["table_rank"]),
            "target":         r["target"],
            "size":           f"{int(len(parts_ordered))}-reg",
            "circuit_score":  round(float(r["circuit_score"]), 2),
            "growth_score":   round(float(r["toxicity"]), 3),
            "combined_score": round(float(r["combined_score"]), 4),
            "part_names":     " → ".join(parts_ordered),
        })
    return pd.DataFrame(rows)


def compute_top5_parts() -> pd.DataFrame:
    """Count part occurrences across the union of top-15 max-circuit
    designs (Table S1) and top-15 Pareto-knee designs (Table S2)."""
    graphs = json.loads((REPO_ROOT / "data" / "topology_g3"
                          / "topology_graphs.json").read_text())
    topo_by_id = {t["topology_id"]: t for t in graphs["topologies"]}

    maxc = _top15_maxcirc_per_target()
    knee = _top15_knee_per_target()

    maxc_counts: Counter = Counter()
    knee_counts: Counter = Counter()
    for _, r in maxc.iterrows():
        for p in _reorder(topo_by_id, r):
            maxc_counts[p] += 1
    for _, r in knee.iterrows():
        for p in _reorder(topo_by_id, r):
            knee_counts[p] += 1

    all_parts = set(maxc_counts) | set(knee_counts)
    totals = [(p, maxc_counts[p], knee_counts[p],
               maxc_counts[p] + knee_counts[p])
              for p in all_parts]
    # Sort by total count desc, then part name asc for ties.
    totals.sort(key=lambda t: (-t[3], t[0]))
    top5 = totals[:5]

    return pd.DataFrame(
        [{"rank": i + 1, "part_name": p,
          "count_in_top15_maxcirc": mc, "count_in_top15_knee": kc,
          "total_count_out_of_30": tot}
         for i, (p, mc, kc, tot) in enumerate(top5)]
    )


def main() -> None:
    df = build_table()
    top5 = compute_top5_parts()

    out = SCRIPT_PATH.parents[1] / "final" / "table_S2.csv"

    # Write the main table, then a blank separator, then a small footer
    # section reporting the top-5 most-frequent parts. Both sections are
    # self-describing so a reader can paste the file into Excel / Word /
    # LaTeX and the footer reads as its own titled sub-table.
    df.to_csv(out, index=False)
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([])   # blank separator row
        w.writerow(["Top-5 most frequent parts across the union of "
                    "Table S1 (top-15 max-circuit) and Table S2 "
                    "(top-15 Pareto-knee) designs"])
        w.writerow(list(top5.columns))
        for _, r in top5.iterrows():
            w.writerow(list(r.values))

    print(f"Wrote: {out.relative_to(REPO_ROOT)}  "
          f"({len(df)} main rows + Top-5 parts footer)")


if __name__ == "__main__":
    main()
