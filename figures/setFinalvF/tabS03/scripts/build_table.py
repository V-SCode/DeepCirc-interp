"""Supplementary Table S3 — Mean circuit-score log₂ enrichment per part × cascade position.

Derived from setFinal/figS14/panel_c (the position × part heatmap).
Rows = 20 library parts (sorted by overall median log_odds, ascending —
most-destructive first, most-constructive last, matching the heatmap).
Columns = 6 cascade-position buckets: NOT-early / NOT-middle / NOT-late /
NOR-early / NOR-middle / NOR-late, where 'late' means depth-to-output = 1
(output-adjacent) and 'early' means d2o >= 3 (input-adjacent).

Values = mean log₂ enrichment (top-5% / bot-5% by circuit score) across
all (part, fp_key) cells that fall into that (part, bucket) combination
after applying the universal-eligibility filter (n_topologies >= 5,
n_targets >= 3, total_designs >= 1000, task = circuit_log).

Empty cells (no fp_key with data for that combination) are left blank
in the CSV — they render as "—" in the source heatmap.

Output:
  final/table_S3.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

L2_TOP05_CSV = (REPO_ROOT / "data" / "topology_g3"
                / "l2_top05" / "l2_enrichment.csv")

BUCKETS = [
    "NOT-early", "NOT-middle", "NOT-late",
    "NOR-early", "NOR-middle", "NOR-late",
]


def bucket_of(row: pd.Series) -> str:
    nt = row["node_type"]
    try:
        d = int(row["depth_to_output"])
    except (TypeError, ValueError):
        d = -1
    if d == 1:
        return f"{nt}-late"
    if d == 2:
        return f"{nt}-middle"
    return f"{nt}-early"


def build_table() -> pd.DataFrame:
    l2 = pd.read_csv(L2_TOP05_CSV)
    elig = l2[
        (l2["task"] == "circuit_log")
        & (l2["n_topologies"] >= 5)
        & (l2["n_targets"] >= 3)
        & (l2["total_designs"] >= 1000)
    ].copy()

    elig["bucket"] = elig.apply(bucket_of, axis=1)

    parts_by_median = (
        elig.groupby("part_name")["log_odds"]
        .median()
        .sort_values()
        .index.tolist()
    )

    grid = (
        elig.groupby(["part_name", "bucket"])["log_odds"]
        .mean()
        .unstack("bucket")
        .reindex(index=parts_by_median, columns=BUCKETS)
        .round(2)
    )
    grid.index.name = "part_name"
    return grid.reset_index()


def main() -> None:
    df = build_table()
    out = SCRIPT_PATH.parents[1] / "final" / "table_S3.csv"
    df.to_csv(out, index=False)
    print(f"Wrote: {out.relative_to(REPO_ROOT)}  ({len(df)} rows × {len(df.columns)-1} bucket cols)")


if __name__ == "__main__":
    main()
