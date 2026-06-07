"""P27 — Size-conditioned typed-variant expansion for non-linear iso classes.

Variant of P25 that computes typed-variant metrics within each size
class S ∈ {4, 5, 6, 7} instead of pooling all topologies.

For each (k, task, size class S, typed motif):

  - parent_iso_key — canonical adjacency string of the typed motif's
    underlying directed graph (collapsing types → ?, lex-smallest over
    all k! permutations). This is what links a typed variant back to
    the parent structural iso class surfaced by P26.
  - within_size_partial_r — partial corr of typed_motif_count vs task
    score over topologies in size S only, controlling for num_edges
    (regress num_edges out of both, correlate residuals).
  - within_size_presence_rate_<tier> — typed-variant presence rate in
    each size-stratified tier (top_01_S / top_05_S / bot_05_S / bot_01_S).

Outputs:
  data/G3/motif_tier_analysis/typed_expansions_by_size_{circuit,toxicity}.csv

User-facing task name "growth" maps to on-disk "toxicity_max".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "topology_g3"

# Reuse helpers from P24 + P25 (numeric-prefix scripts → importlib)
_HERE = Path(__file__).parent
_spec24 = importlib.util.spec_from_file_location(
    "_p24", _HERE / "24_l3_motif_tier_aggregation.py")
_p24 = importlib.util.module_from_spec(_spec24)
sys.modules["_p24"] = _p24
_spec24.loader.exec_module(_p24)
canonical_adj_string     = _p24.canonical_adj_string
partial_r_controlling_for = _p24.partial_r_controlling_for
VIEW_SUFFIX              = _p24.VIEW_SUFFIX

_spec25 = importlib.util.spec_from_file_location(
    "_p25", _HERE / "25_l3_motif_typed_expansion.py")
_p25 = importlib.util.module_from_spec(_spec25)
sys.modules["_p25"] = _p25
_spec25.loader.exec_module(_p25)
parse_typed_motif_edges = _p25.parse_typed_motif_edges


SIZE_CLASSES = (4, 5, 6, 7)
TIERS = ["top_01", "top_05", "top_10", "top_25",
         "bot_25", "bot_10", "bot_05", "bot_01"]
TASKS = ["circuit", "toxicity"]
SCORE_FOR_TASK = {"circuit": "circuit_log_max", "toxicity": "toxicity_max"}

LOG2 = np.log(2.0)


def expand_for_k(k: int, motif_view: str = "full") -> dict[str, pd.DataFrame]:
    """Per (size, task, typed motif) row, with within-size metrics.

    ``motif_view`` mirrors P13/P24/P26 ('full', 'no_in', 'internal_only').
    """
    k_suffix    = "" if k == 3 else f"_k{k}"
    view_suffix = VIEW_SUFFIX[motif_view]

    # Per-topology typed motif counts (wide)
    counts = pd.read_csv(
        DATA / "l3"
        / f"l3_motif_counts_typed{k_suffix}{view_suffix}.csv"
    )
    typed_motifs = [c for c in counts.columns if c != "topology_id"]

    # Canonical adj string per typed motif (links to parent iso class)
    parent_iso_of: dict[str, str] = {}
    for m in typed_motifs:
        k_m, adj = parse_typed_motif_edges(m)
        if k_m != k:
            raise ValueError(f"k mismatch for {m}")
        parent_iso_of[m] = canonical_adj_string(adj)

    # Size-stratified tier counts
    tier = pd.read_csv(DATA / "size_tier_designs" / "size_tier_design_counts.csv")
    merged = counts[["topology_id"]].merge(tier, on="topology_id", how="left")
    present_mask = merged.notna().all(axis=1).to_numpy()
    n_dropped = int((~present_mask).sum())
    if n_dropped > 0:
        print(f"  [k={k}] dropping {n_dropped} topologies without tier data")
    merged = merged[present_mask].reset_index(drop=True)
    counts = counts[present_mask].reset_index(drop=True)

    # Topology features
    feats_full = pd.read_csv(DATA / "l3" / "l3_topology_features.csv")
    feats = counts[["topology_id"]].merge(feats_full, on="topology_id", how="left")
    if feats.isna().any().any():
        raise SystemExit("missing topology features for some topology_ids")
    num_edges_topo = feats["num_edges"].to_numpy().astype(float)
    regulator_count = feats["regulator_count"].to_numpy().astype(int)

    typed_mat = counts[typed_motifs].fillna(0).to_numpy().astype(np.int64)
    typed_present = (typed_mat > 0).astype(np.int64)

    out: dict[str, pd.DataFrame] = {}
    for task in TASKS:
        score_vec = feats[SCORE_FOR_TASK[task]].to_numpy().astype(float)
        rows = []
        for s in SIZE_CLASSES:
            mask_s = (regulator_count == s)
            if mask_s.sum() == 0:
                continue
            score_s = score_vec[mask_s]
            num_edges_s = num_edges_topo[mask_s]
            typed_count_s = typed_mat[mask_s, :].astype(float)
            typed_present_s = typed_present[mask_s, :]

            # Within-size tier counts. Skip tiers whose column is missing
            # from the input CSV (defensive — top_10/bot_10 columns appear
            # only after P23.2 re-run.)
            tier_total: dict[str, int] = {}
            presence_designs: dict[str, np.ndarray] = {}
            for t in TIERS:
                col = f"n_{t}_{task}_S"
                if col not in merged.columns:
                    continue
                n_in_tier_full = merged[col].to_numpy().astype(np.int64)
                n_in_tier = np.where(mask_s, n_in_tier_full, 0)
                tier_total[t] = int(n_in_tier.sum())
                presence_designs[t] = typed_present.T @ n_in_tier

            # partial r per typed motif (within size)
            n_typed = len(typed_motifs)
            partial_r_per_typed = np.full(n_typed, np.nan, dtype=float)
            if typed_count_s.shape[0] >= 3:
                for j in range(n_typed):
                    partial_r_per_typed[j] = partial_r_controlling_for(
                        typed_count_s[:, j], score_s, num_edges_s)

            for j, motif in enumerate(typed_motifs):
                n_topo_size_with = int(typed_present_s[:, j].sum())
                if n_topo_size_with == 0:
                    # Typed variant doesn't appear at this size; skip emit.
                    continue
                row: dict = {
                    "k": k,
                    "task": task,
                    "size_class": s,
                    "parent_iso_key": parent_iso_of[motif],
                    "typed_motif": motif,
                    "n_topo_in_size": int(mask_s.sum()),
                    "n_topo_with_motif_in_size": n_topo_size_with,
                    "partial_r": (float(partial_r_per_typed[j])
                                    if np.isfinite(partial_r_per_typed[j])
                                    else None),
                }
                for t in TIERS:
                    if t not in presence_designs:
                        continue
                    pres = int(presence_designs[t][j])
                    tot = tier_total[t]
                    rate = pres / tot if tot > 0 else float("nan")
                    row[f"presence_rate_{t}"] = rate
                    row[f"presence_designs_{t}"] = pres
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(["size_class", "parent_iso_key", "typed_motif"]).reset_index(drop=True)
        out[task] = df
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", default="G3")
    p.add_argument("--motif_view",
                   choices=list(VIEW_SUFFIX.keys()),
                   default="full",
                   help="Motif-view axis (mirrors P13/P24/P26). 'full' "
                        "(default), 'no_in' (sensor IN stripped), or "
                        "'internal_only' (both IN and OUT-* stripped). "
                        "Outputs gain a matching '_no_in' / '_internal_only' "
                        "suffix.")
    args = p.parse_args()
    if args.group != "G3":
        raise SystemExit("only --group G3 supported")

    view_suffix = VIEW_SUFFIX[args.motif_view]
    out_dir = DATA / "motif_tier_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"phase": "27_l3_motif_size_typed_expansion",
                       "motif_view": args.motif_view,
                       "size_classes": list(SIZE_CLASSES),
                       "per_k_summary": {}}
    full_circuit = []
    full_toxicity = []
    for k in (3, 4, 5):
        print(f"[k={k}] size-conditioned typed expansion "
              f"(view={args.motif_view})...")
        tables = expand_for_k(k, motif_view=args.motif_view)
        for task, df in tables.items():
            print(f"  task={task:9s}  n_rows={len(df):5d}  "
                  f"n_typed_motifs={df['typed_motif'].nunique():4d}  "
                  f"n_parent_iso={df['parent_iso_key'].nunique():3d}")
        full_circuit.append(tables["circuit"])
        full_toxicity.append(tables["toxicity"])
        summary["per_k_summary"][f"k={k}"] = {
            "n_circuit_rows": int(len(tables["circuit"])),
            "n_toxicity_rows": int(len(tables["toxicity"])),
        }

    circuit_all = pd.concat(full_circuit, ignore_index=True)
    toxicity_all = pd.concat(full_toxicity, ignore_index=True)
    out_circ = (out_dir
                / f"typed_expansions_by_size_circuit{view_suffix}.csv")
    out_tox  = (out_dir
                / f"typed_expansions_by_size_toxicity{view_suffix}.csv")
    circuit_all.to_csv(out_circ, index=False)
    toxicity_all.to_csv(out_tox,  index=False)
    print(f"\nwrote {out_circ.relative_to(REPO)}  ({len(circuit_all)} rows)")
    print(f"wrote {out_tox.relative_to(REPO)}  ({len(toxicity_all)} rows)")

    out_sum = (out_dir
               / f"typed_expansions_by_size_summary{view_suffix}.json")
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_sum.relative_to(REPO)}")


if __name__ == "__main__":
    main()
