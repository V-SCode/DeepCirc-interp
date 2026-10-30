"""Table S3: mean circuit-presence enrichment by gate type and depth.

Each value is an unweighted mean of context-specific log2 probability ratios.
Each context denominator counts circuits once, regardless of repeated matching
positions. Historical top/bottom score cutoffs and eligibility are preserved.
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

from figures.setFinalvF._shared.presence_data import (
    load_presence_data, VALUE_COLUMN,
)

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
    elig = load_presence_data("circuit_log")

    elig["bucket"] = elig.apply(bucket_of, axis=1)

    parts_by_median = (
        elig.groupby("part_name")[VALUE_COLUMN]
        .median()
        .sort_values()
        .index.tolist()
    )

    grid = (
        elig.groupby(["part_name", "bucket"])[VALUE_COLUMN]
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
