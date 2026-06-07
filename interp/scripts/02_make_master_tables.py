#!/usr/bin/env python3
"""For each exemplar, build a master parquet table with one row per unique permutation:

    permutation (tuple, also stored as 'perm_str' for pandas hashing)
    circuit_score_sim          -- simulator ground truth (NaN if missing)
    growth_score_sim           -- simulator ground truth (NaN if missing)
    interference_flag_sim      -- bool
    circuit_score_pred         -- MLP prediction
    growth_score_pred          -- MLP prediction
    source_training            -- was this perm in the training-data pickle set?
    source_nn_high             -- was this perm in the NN-predicted-high set?
    source_top_ranked          -- was this perm in scores_top_designs_*.csv?
    is_yellow_dot              -- sim-rank-1 design indicator
    rank                       -- rank in top-ranked CSV (pd.NA if not in top-ranked)

Write to: processed/master_{name}.parquet
Also emits: processed/summary_{name}.json (row counts, merge coverage stats).

We DO NOT re-run the simulator — ground-truth scores only exist where already labeled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders as L

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def perm_str(t) -> str:
    return "-".join(str(int(x)) for x in t)


def build_master(name: str) -> pd.DataFrame:
    ex = L.EXEMPLARS[name]

    # 1. training-data labels
    td = L.load_training_data(name)
    td["source"] = "training"

    # 2. scored-high-predicted labels (NN-picked, then sim-rescored)
    shp = L.load_scored_high_predicted(name)
    shp["source"] = "nn_high"

    # 3. predicted-high-perms npy — extra perms (no sim scores) that we may want to know about
    #    but per our inspection, len(shp) == len(npy), so these are the same set.
    npy = L.load_predicted_high_perms(name)
    assert len(npy) == len(shp), f"{name}: npy len {len(npy)} != shp len {len(shp)}"

    # 4. top-ranked CSV — sim-rank within nn_high. We'll merge its rank onto shp.
    top = L.load_top_ranked(name)
    top["rank"] = np.arange(1, len(top) + 1, dtype=np.int64)

    # Concatenate labeled sources (training + nn_high), dedup by perm, keeping first sim value
    labeled = pd.concat([td[["permutation","circuit_score","growth_score","interference_flag","source"]],
                         shp[["permutation","circuit_score","growth_score","interference_flag","source"]]],
                        ignore_index=True)
    labeled["perm_str"] = labeled["permutation"].apply(perm_str)

    # Pivot sources into boolean columns
    labeled["source_training"] = labeled["source"].eq("training")
    labeled["source_nn_high"] = labeled["source"].eq("nn_high")
    agg = (labeled.groupby("perm_str", as_index=False)
                    .agg({
                        "permutation": "first",
                        "circuit_score": "first",
                        "growth_score": "first",
                        "interference_flag": "first",
                        "source_training": "max",
                        "source_nn_high": "max",
                    }))
    agg = agg.rename(columns={"circuit_score": "circuit_score_sim",
                              "growth_score": "growth_score_sim",
                              "interference_flag": "interference_flag_sim"})

    # Merge top-rank info
    top["perm_str"] = top["permutation"].apply(perm_str)
    agg = agg.merge(top[["perm_str", "rank"]], on="perm_str", how="left")
    agg["source_top_ranked"] = agg["rank"].notna()
    agg["is_yellow_dot"] = agg["rank"].eq(1)

    # MLP predictions for all rows
    cm = L.load_circuit_mlp(name)
    gm = L.load_growth_mlp(name)
    perms = np.stack([np.asarray(p, dtype=np.int64) for p in agg["permutation"]])
    agg["circuit_score_pred"] = L.predict(cm, perms).astype(np.float64)
    agg["growth_score_pred"] = L.predict(gm, perms).astype(np.float64)

    # Reorder columns
    cols = ["perm_str", "permutation",
            "circuit_score_sim", "growth_score_sim", "interference_flag_sim",
            "circuit_score_pred", "growth_score_pred",
            "source_training", "source_nn_high", "source_top_ranked",
            "rank", "is_yellow_dot"]
    return agg[cols]


def main():
    summary = {}
    for name in L.EXEMPLARS:
        print(f"\n[{name}] building master table ...")
        m = build_master(name)
        out = PROC / f"master_{name}.parquet"
        # permutation tuples can't be stored directly; convert to list
        m_to_save = m.copy()
        m_to_save["permutation"] = m_to_save["permutation"].apply(list)
        m_to_save.to_parquet(out, index=False)
        print(f"  wrote {out} ({len(m):,} rows)")

        s = {
            "n_rows": int(len(m)),
            "n_training": int(m.source_training.sum()),
            "n_nn_high": int(m.source_nn_high.sum()),
            "n_top_ranked": int(m.source_top_ranked.sum()),
            "n_both_training_and_nn_high": int((m.source_training & m.source_nn_high).sum()),
            "n_yellow_dot": int(m.is_yellow_dot.sum()),
            "pred_circuit_mean": float(m.circuit_score_pred.mean()),
            "pred_growth_mean": float(m.growth_score_pred.mean()),
            "sim_circuit_mean": float(m.circuit_score_sim.mean()),
            "sim_growth_mean": float(m.growth_score_sim.mean()),
            "interference_frac": float(m.interference_flag_sim.mean()),
        }
        summary[name] = s
        print(f"  summary: {s}")
        with open(PROC / f"summary_{name}.json", "w") as f:
            json.dump(s, f, indent=2)

    with open(PROC / "summary_all.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nmaster tables built for {list(summary)}; combined summary at {PROC/'summary_all.json'}")


if __name__ == "__main__":
    main()
