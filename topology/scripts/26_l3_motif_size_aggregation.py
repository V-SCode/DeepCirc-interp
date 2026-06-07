"""P26 — Size-conditioned non-linear motif rankings.

Reuses P24's iso canonicalization mapping and applies the same three
metrics (partial r, matched-tail log_odds, per-design presence rate)
within each size class S ∈ {4, 5, 6, 7} instead of pooling across the
full population.

For each (k, task, size class S, iso class):

  - presence_rate_in_S_in_tier =
        Σ_T∈S [iso_present(T) × n_designs(T, in tier_S)]
            / total_n_designs(S, in tier_S)
    where the tier is size-stratified (n_top_01_circuit_S etc. from
    P23.2's `size_tier_design_counts.csv`).
  - matched-tail log_odds within S:
        log_odds_01_S = log2(presence_rate_top_01_S / presence_rate_bot_01_S)
        log_odds_05_S = log2(presence_rate_top_05_S / presence_rate_bot_05_S)
  - partial r within S: partial corr of iso_class_count and score,
    over the topologies in size class S only, controlling for num_edges
    (regress num_edges out of both, correlate residuals).
  - linearity flag inherited from P24.

Outputs one CSV per (k, task) pair under
  data/G3/motif_tier_analysis/structural_iso_by_size_{circuit,toxicity}_k{3,4,5}.csv
plus a summary JSON.

User-facing task name "growth" maps to on-disk "toxicity_max" per the
data_schema memory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "topology_g3"

# Reuse parse_motif + canonical_adj_string + is_linear_iso + partial_r helper
# from P24 (numeric-prefix scripts can't be normal imports).
_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "_p24", _HERE / "24_l3_motif_tier_aggregation.py")
_p24 = importlib.util.module_from_spec(_spec)
sys.modules["_p24"] = _p24
_spec.loader.exec_module(_p24)
parse_motif              = _p24.parse_motif
canonical_adj_string     = _p24.canonical_adj_string
is_linear_iso            = _p24.is_linear_iso
partial_r_controlling_for = _p24.partial_r_controlling_for
VIEW_SUFFIX              = _p24.VIEW_SUFFIX


SIZE_CLASSES = (4, 5, 6, 7)
# Matched-tail tier pairs available downstream. Each pair (top_<N>, bot_<N>)
# becomes a `log_odds_top<N>_vs_bot<N>` column in P26's output.
TIERS = ["top_01", "top_05", "top_10", "top_25",
         "bot_25", "bot_10", "bot_05", "bot_01"]
TIER_PAIRS = [("01", "01"), ("05", "05"), ("10", "10"), ("25", "25")]
TASKS = ["circuit", "toxicity"]
SCORE_FOR_TASK = {"circuit": "circuit_log_max", "toxicity": "toxicity_max"}

LOG2 = np.log(2.0)


def aggregate_for_k(k: int, motif_view: str = "full") -> dict[str, pd.DataFrame]:
    """Aggregate per-(size, task) iso-class metrics for one k.

    Returns {task: rankings_df} with one row per (size_class, iso_key).

    ``motif_view`` mirrors the P13 / P24 view axis ('full', 'no_in',
    'internal_only') so the size-conditioned re-cut can be done on any
    of the three motif universes.
    """
    k_suffix    = "" if k == 3 else f"_k{k}"
    view_suffix = VIEW_SUFFIX[motif_view]

    # Raw motif counts per topology (wide)
    counts = pd.read_csv(
        DATA / "l3"
        / f"l3_motif_counts_structural{k_suffix}{view_suffix}.csv"
    )
    raw_motifs = [c for c in counts.columns if c != "topology_id"]

    # Canonicalize each raw motif → iso_key (matches P24)
    iso_key_of: dict[str, str] = {}
    for m in raw_motifs:
        _, adj = parse_motif(m)
        iso_key_of[m] = canonical_adj_string(adj)
    iso_keys = sorted(set(iso_key_of.values()))
    iso_idx_of_key = {key: i for i, key in enumerate(iso_keys)}

    # Iso-class-summed counts (n_topo, n_iso)
    raw_mat = counts[raw_motifs].fillna(0).to_numpy().astype(np.int64)
    n_topo = raw_mat.shape[0]
    iso_mat = np.zeros((n_topo, len(iso_keys)), dtype=np.int64)
    for j, m in enumerate(raw_motifs):
        ik = iso_idx_of_key[iso_key_of[m]]
        iso_mat[:, ik] += raw_mat[:, j]
    iso_present = (iso_mat > 0).astype(np.int64)

    # Size-stratified tier counts (P23.2 output)
    tier = pd.read_csv(DATA / "size_tier_designs" / "size_tier_design_counts.csv")
    merged = counts[["topology_id"]].merge(tier, on="topology_id", how="left")
    if merged.isna().any().any():
        # Some topologies in counts have no tier data — usually A0/A3 markers.
        # Drop them from the analysis.
        present_mask = merged.notna().all(axis=1).to_numpy()
        n_dropped = int((~present_mask).sum())
        print(f"  [k={k}] dropping {n_dropped} topologies without tier data")
        merged = merged[present_mask].reset_index(drop=True)
        raw_mat = raw_mat[present_mask]
        iso_mat = iso_mat[present_mask]
        iso_present = iso_present[present_mask]
        counts = counts[present_mask].reset_index(drop=True)

    # Topology features for partial_r recompute
    feats_full = pd.read_csv(DATA / "l3" / "l3_topology_features.csv")
    feats = counts[["topology_id"]].merge(feats_full, on="topology_id", how="left")
    if feats.isna().any().any():
        raise SystemExit("missing topology features for some topology_ids")
    num_edges_topo = feats["num_edges"].to_numpy().astype(float)
    regulator_count = feats["regulator_count"].to_numpy().astype(int)

    out: dict[str, pd.DataFrame] = {}
    for task in TASKS:
        score_vec = feats[SCORE_FOR_TASK[task]].to_numpy().astype(float)
        rows = []
        for s in SIZE_CLASSES:
            mask_s = (regulator_count == s)
            n_topo_s = int(mask_s.sum())
            if n_topo_s == 0:
                continue

            # Per-tier total designs in this size class. Skip tiers whose
            # column is missing from the input CSV — useful while P23.2's
            # new top_10/bot_10/top_25/bot_25 columns are still propagating.
            tier_total: dict[str, int] = {}
            presence_designs: dict[str, np.ndarray] = {}
            n_topo_in_tier: dict[str, np.ndarray] = {}
            for t in TIERS:
                col = f"n_{t}_{task}_S"
                if col not in merged.columns:
                    continue
                n_in_tier_full = merged[col].to_numpy().astype(np.int64)
                n_in_tier = np.where(mask_s, n_in_tier_full, 0)
                tier_total[t] = int(n_in_tier.sum())
                presence_designs[t] = iso_present.T @ n_in_tier
                mask_topo_in_tier = (n_in_tier > 0)
                n_topo_in_tier[t] = (
                    (iso_present > 0)
                    & mask_topo_in_tier[:, None]
                ).sum(axis=0)

            # partial r within size class (recompute per iso class)
            iso_count_s = iso_mat[mask_s, :].astype(float)
            score_s = score_vec[mask_s]
            num_edges_s = num_edges_topo[mask_s]
            partial_r_per_iso = np.full(len(iso_keys), np.nan, dtype=float)
            for j in range(len(iso_keys)):
                if iso_count_s.shape[0] >= 3:  # need ≥3 topos for a meaningful r
                    partial_r_per_iso[j] = partial_r_controlling_for(
                        iso_count_s[:, j], score_s, num_edges_s)

            for j, iso_key in enumerate(iso_keys):
                num_edges_motif = int(sum(int(b) for b in iso_key))
                adj_mtx = np.array([int(b) for b in iso_key],
                                     dtype=np.int8).reshape(k, k)
                row: dict = {
                    "k": k,
                    "task": task,
                    "size_class": s,
                    "iso_key": iso_key,
                    "n_topo_in_size": n_topo_s,
                    "n_topo_with_motif_in_size": int(iso_present[mask_s, j].sum()),
                    "num_edges": num_edges_motif,
                    "max_in_deg": int(adj_mtx.sum(axis=0).max()),
                    "max_out_deg": int(adj_mtx.sum(axis=1).max()),
                    "is_linear": is_linear_iso(iso_key, k),
                    "partial_r": float(partial_r_per_iso[j])
                                    if np.isfinite(partial_r_per_iso[j]) else None,
                }
                for t in TIERS:
                    if t not in presence_designs:
                        continue
                    pres = int(presence_designs[t][j])
                    tot = tier_total[t]
                    rate = pres / tot if tot > 0 else float("nan")
                    row[f"presence_rate_{t}"] = rate
                    row[f"presence_designs_{t}"] = pres
                    row[f"n_topo_{t}"] = int(n_topo_in_tier[t][j])

                def lo(a: str, b: str) -> float:
                    ra = row.get(f"presence_rate_{a}")
                    rb = row.get(f"presence_rate_{b}")
                    if (ra is None or rb is None
                            or not np.isfinite(ra) or not np.isfinite(rb)
                            or ra <= 0 or rb <= 0):
                        return float("nan")
                    return float(np.log(ra / rb) / LOG2)
                for tp, bp in TIER_PAIRS:
                    # Only emit the log_odds column if both tier columns exist.
                    if (f"top_{tp}" in tier_total
                            and f"bot_{bp}" in tier_total):
                        row[f"log_odds_top{tp}_vs_bot{bp}"] = lo(
                            f"top_{tp}", f"bot_{bp}")
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(["size_class", "is_linear", "iso_key"]).reset_index(drop=True)
        out[task] = df
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", default="G3")
    p.add_argument("--motif_view",
                   choices=list(VIEW_SUFFIX.keys()),
                   default="full",
                   help="Motif-view axis (mirrors P13/P24). 'full' (default), "
                        "'no_in' (sensor IN stripped), 'no_out' (OUT-* "
                        "stripped), or 'internal_only' (both IN and OUT-* "
                        "stripped). Outputs gain a matching filename suffix.")
    p.add_argument("--min_n_topo", type=int, default=1,
                   help="Minimum n_topo_with_motif_in_size for an iso class "
                        "to be retained per (size, task). Default 1 (no "
                        "filter) — surfaces every iso class with finite "
                        "log_odds. Set to 5 to reproduce the historical "
                        "P13 --min_motif_topologies 5 behaviour at iso level.")
    args = p.parse_args()
    if args.group != "G3":
        raise SystemExit("only --group G3 supported (P23.2 outputs are G3-only)")

    view_suffix = VIEW_SUFFIX[args.motif_view]
    out_dir = DATA / "motif_tier_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "phase": "26_l3_motif_size_aggregation",
        "motif_view": args.motif_view,
        "min_n_topo": args.min_n_topo,
        "size_classes": list(SIZE_CLASSES),
        "per_k_summary": {},
    }
    for k in (3, 4, 5):
        print(f"[k={k}] size-conditioned iso aggregation "
              f"(view={args.motif_view}, min_n_topo={args.min_n_topo})...")
        tables = aggregate_for_k(k, motif_view=args.motif_view)
        for task, df in tables.items():
            # Apply iso-level n_topo filter (per-size). Default 1 = no filter.
            if args.min_n_topo > 1:
                before = len(df)
                df = df[df["n_topo_with_motif_in_size"] >= args.min_n_topo
                        ].reset_index(drop=True)
                print(f"    [filter] n_topo >= {args.min_n_topo}: "
                      f"{before} -> {len(df)} rows")
            n_rows = len(df)
            n_iso = df["iso_key"].nunique()
            n_linear = int(df[df["size_class"] == SIZE_CLASSES[0]]["is_linear"].sum())
            print(f"  task={task:9s}  n_rows={n_rows:4d}  n_iso={n_iso:3d}  "
                  f"n_linear={n_linear} (per size class)")
            out_path = (out_dir
                        / f"structural_iso_by_size_{task}_k{k}"
                          f"{view_suffix}.csv")
            df.to_csv(out_path, index=False)
            print(f"  wrote {out_path.relative_to(REPO)}")
        summary["per_k_summary"][f"k={k}"] = {
            "n_rows_per_task": int(len(tables["circuit"])),
            "n_iso_classes":   int(tables["circuit"]["iso_key"].nunique()),
        }

    out_sum = (out_dir
               / f"structural_iso_by_size_summary{view_suffix}.json")
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_sum.relative_to(REPO)}")


if __name__ == "__main__":
    main()
