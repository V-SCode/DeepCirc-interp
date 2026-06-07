"""P24 — Structural motif × tier rankings for simplemotif analysis.

The raw motif IDs from P13 are NOT isomorphism-canonical (different node
orderings of the same shape produce different motif IDs). This script
canonicalizes by enumerating all k! node permutations and taking the
lexicographically smallest adjacency string per motif, then aggregates
raw motifs into iso-classes.

For each k ∈ {3, 4, 5} and task ∈ {circuit, toxicity}:

  - iso_class_count(T) = Σ_{m ∈ class} motif_count(m, T)
  - iso_class_present(T) = 1 iff iso_class_count(T) > 0
  - presence_rate_in_tier(class, T) =
        Σ_T [iso_class_present(T) × n_designs(T, in tier)]
            / total_n_designs_in_tier
    (matches fig23's per-design presence rate convention).
  - log_odds matched-tail:
        log_odds_01 = log2(presence_rate_top_01 / presence_rate_bot_01)
        log_odds_05 = log2(presence_rate_top_05 / presence_rate_bot_05)
  - partial_r — recomputed per iso-class against iso_class_count,
    controlling for num_edges of the topology (NOT the motif). This is
    the same convention P13 uses for raw motifs.
  - linearity flag: an iso-class is linear iff its canonical adjacency
    is a directed path P_k (num_edges == k-1 AND max in/out deg ≤ 1
    over the motif itself).

Outputs one CSV per (k, task) under
  data/G3/motif_tier_analysis/structural_iso_rankings_{circuit,growth}_k{3,4,5}.csv
plus a summary JSON, plus a raw-to-iso mapping CSV per k.

The user-facing task name "growth" maps to the on-disk task name
"toxicity_max" (paper alias: growth_score = toxicity_score).
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "topology_g3"

TIERS = ["top_01", "top_05", "bot_05", "bot_01"]
TASKS = ["circuit", "toxicity"]
SCORE_FOR_TASK = {"circuit": "circuit_log_max", "toxicity": "toxicity_max"}

LOG2 = np.log(2.0)

# Motif-view → P13 output-filename suffix mapping. Must mirror P13's own
# VIEW_SUFFIX table (13_l3_motif_features.py).
VIEW_SUFFIX = {"full": "", "no_in": "_no_in",
               "no_out": "_no_out",
               "internal_only": "_internal_only"}


# ---------------------------------------------------------------------------
# Motif parsing + canonical iso form
# ---------------------------------------------------------------------------

def parse_motif(motif: str) -> tuple[int, np.ndarray]:
    """Parse a structural motif id into (k, adjacency matrix int8[k, k])."""
    types_part, edges_part = motif.split("|edges=")
    k = len(types_part.split("/"))
    if len(edges_part) != k * k:
        raise ValueError(f"edge bits {len(edges_part)} != k*k = {k*k}")
    bits = np.array([int(b) for b in edges_part], dtype=np.int8)
    return k, bits.reshape(k, k)


def canonical_adj_string(adj: np.ndarray) -> str:
    """Lex-smallest adjacency string over all node permutations.

    For k ≤ 5 this is 120 permutations max — fast.
    """
    k = adj.shape[0]
    best = None
    for perm in permutations(range(k)):
        perm = list(perm)
        sub = adj[np.ix_(perm, perm)]
        s = "".join(str(int(x)) for x in sub.flatten())
        if best is None or s < best:
            best = s
    return best  # type: ignore[return-value]


def is_linear_iso(canon_str: str, k: int) -> bool:
    """Check whether the canonical adjacency string is a directed path P_k."""
    adj = np.array([int(b) for b in canon_str], dtype=np.int8).reshape(k, k)
    num_edges = int(adj.sum())
    if num_edges != k - 1:
        return False
    in_deg = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    return bool(in_deg.max() <= 1 and out_deg.max() <= 1)


# ---------------------------------------------------------------------------
# Partial-r recompute (regress num_edges out of both motif count and score)
# ---------------------------------------------------------------------------

def partial_r_controlling_for(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation of x, y after regressing out z."""
    def resid(v: np.ndarray, z_c: np.ndarray, z_var: float) -> np.ndarray:
        slope = float((v * z_c).sum() / z_var) if z_var > 1e-12 else 0.0
        return v - slope * z_c - v.mean()
    z_c = z - z.mean()
    z_var = float((z_c ** 2).sum())
    rx = resid(x.astype(float), z_c, z_var)
    ry = resid(y.astype(float), z_c, z_var)
    num = float((rx * ry).sum())
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return num / denom if denom > 1e-12 else float("nan")


# ---------------------------------------------------------------------------
# Per-k aggregation
# ---------------------------------------------------------------------------

def aggregate_for_k(k: int,
                    motif_view: str = "full"
                    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Aggregate iso-class metrics for one k.

    Returns ({task: rankings_df}, raw_to_iso_mapping_df).

    ``motif_view`` selects which P13 motif-count CSV to read (mirrors the
    view-suffix convention from P13).
    """
    k_suffix    = "" if k == 3 else f"_k{k}"
    view_suffix = VIEW_SUFFIX[motif_view]

    counts = pd.read_csv(
        DATA / "l3"
        / f"l3_motif_counts_structural{k_suffix}{view_suffix}.csv"
    )
    raw_motifs = [c for c in counts.columns if c != "topology_id"]

    # Canonicalize each raw motif id → iso_key (canonical adj string).
    iso_key_of: dict[str, str] = {}
    for m in raw_motifs:
        _, adj = parse_motif(m)
        iso_key_of[m] = canonical_adj_string(adj)
    iso_keys = sorted(set(iso_key_of.values()))
    iso_idx_of_key = {key: i for i, key in enumerate(iso_keys)}

    # Build iso-class-summed counts per topology: (n_topo, n_iso).
    raw_mat = counts[raw_motifs].fillna(0).to_numpy().astype(np.int64)
    n_topo = raw_mat.shape[0]
    iso_mat = np.zeros((n_topo, len(iso_keys)), dtype=np.int64)
    for j, m in enumerate(raw_motifs):
        ik = iso_idx_of_key[iso_key_of[m]]
        iso_mat[:, ik] += raw_mat[:, j]
    iso_present = (iso_mat > 0).astype(np.int64)

    # Per-topo tier counts.
    tier = pd.read_csv(DATA / "global_tier_designs" / "global_tier_design_counts.csv")
    merged = counts[["topology_id"]].merge(tier, on="topology_id", how="left")
    if merged.isna().any().any():
        raise SystemExit(f"missing tier-count rows for some topology_ids at k={k}")

    # Per-topology features (for num_edges + score).
    feats = pd.read_csv(DATA / "l3" / "l3_topology_features.csv")
    feats = counts[["topology_id"]].merge(feats, on="topology_id", how="left")
    if feats.isna().any().any():
        raise SystemExit("missing topology features for some topology_ids")
    num_edges_topo = feats["num_edges"].to_numpy().astype(float)

    out: dict[str, pd.DataFrame] = {}
    raw_to_iso_rows: list[dict] = []
    for raw_m, iso_key in iso_key_of.items():
        raw_to_iso_rows.append({"k": k, "raw_motif": raw_m,
                                  "iso_key": iso_key})

    for task in TASKS:
        score_vec = feats[SCORE_FOR_TASK[task]].to_numpy().astype(float)
        tier_total = {}
        presence_designs = {}
        n_topo_in_tier = {}
        for t in TIERS:
            col = f"n_{t}_{task}"
            n_in_tier = merged[col].to_numpy().astype(np.int64)
            tier_total[t] = int(n_in_tier.sum())
            presence_designs[t] = iso_present.T @ n_in_tier
            mask_topo_in_tier = n_in_tier > 0
            n_topo_in_tier[t] = ((iso_present > 0)
                                    & mask_topo_in_tier[:, None]).sum(axis=0)

        rows = []
        for i, iso_key in enumerate(iso_keys):
            class_count = iso_mat[:, i].astype(float)
            partial_r = partial_r_controlling_for(class_count, score_vec,
                                                    num_edges_topo)
            num_edges_motif = int(sum(int(b) for b in iso_key))
            adj_mtx = np.array([int(b) for b in iso_key], dtype=np.int8).reshape(k, k)
            row = {
                "k": k,
                "task": task,
                "iso_key": iso_key,
                "n_iso_members": int(sum(1 for v in iso_key_of.values()
                                          if v == iso_key)),
                "num_edges": num_edges_motif,
                "max_in_deg": int(adj_mtx.sum(axis=0).max()),
                "max_out_deg": int(adj_mtx.sum(axis=1).max()),
                "is_linear": is_linear_iso(iso_key, k),
                "partial_r": partial_r,
                "n_topo_total": int(iso_present[:, i].sum()),
            }
            for t in TIERS:
                pres = int(presence_designs[t][i])
                tot = tier_total[t]
                rate = pres / tot if tot > 0 else float("nan")
                row[f"presence_rate_{t}"] = rate
                row[f"presence_designs_{t}"] = pres
                row[f"n_topo_{t}"] = int(n_topo_in_tier[t][i])

            def lo(a: str, b: str) -> float:
                ra = row[f"presence_rate_{a}"]
                rb = row[f"presence_rate_{b}"]
                if (ra is None or rb is None
                        or not np.isfinite(ra) or not np.isfinite(rb)
                        or ra <= 0 or rb <= 0):
                    return float("nan")
                return float(np.log(ra / rb) / LOG2)
            row["log_odds_top01_vs_bot01"] = lo("top_01", "bot_01")
            row["log_odds_top05_vs_bot05"] = lo("top_05", "bot_05")
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(["is_linear", "iso_key"]).reset_index(drop=True)
        df.attrs["tier_total"] = tier_total
        out[task] = df

    raw_to_iso_df = pd.DataFrame(raw_to_iso_rows)
    return out, raw_to_iso_df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", default="G3")
    p.add_argument("--motif_view",
                   choices=list(VIEW_SUFFIX.keys()),
                   default="full",
                   help="P13 motif-count view to consume. 'full' reads "
                        "l3_motif_counts_structural*.csv (default). 'no_in' "
                        "reads *_no_in.csv (sensor IN nodes stripped). "
                        "'internal_only' reads *_internal_only.csv (both IN "
                        "and OUT-* stripped; only internal NOT + NOR gates). "
                        "Output filenames gain the same suffix.")
    args = p.parse_args()

    if args.group != "G3":
        raise SystemExit("only --group G3 supported (P23 outputs only exist for G3)")

    view_suffix = VIEW_SUFFIX[args.motif_view]
    out_dir = DATA / "motif_tier_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "phase": "24_l3_motif_tier_aggregation",
        "motif_view": args.motif_view,
        "tier_pool_sizes": {},
        "per_k_summary": {},
    }
    for k in (3, 4, 5):
        print(f"[k={k}] canonicalizing + aggregating structural motifs "
              f"(view={args.motif_view})...")
        tables, raw_iso_df = aggregate_for_k(k, motif_view=args.motif_view)
        raw_iso_path = (out_dir
                        / f"raw_to_iso_mapping_k{k}{view_suffix}.csv")
        raw_iso_df.to_csv(raw_iso_path, index=False)
        print(f"  wrote {raw_iso_path.relative_to(REPO)}")

        for task, df in tables.items():
            n_linear = int(df["is_linear"].sum())
            n_total = len(df)
            n_raw = int(raw_iso_df["raw_motif"].nunique())
            print(f"  task={task:9s}  n_iso={n_total:3d}  n_raw={n_raw:3d}  "
                  f"n_linear_iso={n_linear}")
            out_path = (out_dir
                        / f"structural_iso_rankings_{task}_k{k}"
                          f"{view_suffix}.csv")
            df.to_csv(out_path, index=False)
            print(f"  wrote {out_path.relative_to(REPO)}")
        sample_df = tables["circuit"]
        if "tier_total" in sample_df.attrs:
            summary["tier_pool_sizes"]["circuit"] = sample_df.attrs["tier_total"]
            summary["tier_pool_sizes"]["toxicity"] = (
                tables["toxicity"].attrs["tier_total"])
        summary["per_k_summary"][f"k={k}"] = {
            "n_iso_classes": int(len(tables["circuit"])),
            "n_linear_iso": int(tables["circuit"]["is_linear"].sum()),
            "n_raw_motifs": int(raw_iso_df["raw_motif"].nunique()),
        }
    out_sum = (out_dir
               / f"structural_iso_rankings_summary{view_suffix}.json")
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_sum.relative_to(REPO)}")


if __name__ == "__main__":
    main()
