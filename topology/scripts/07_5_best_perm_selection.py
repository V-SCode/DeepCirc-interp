"""Phase 7.5 — best-perm-per-topology selection under three criteria.

For each A1+A2 topology, this phase computes which input-permutation is the
"best" representative under three different definitions and reports their
agreement. The motivation is to derisk our default Definition-A convention
(used by P5/P9/P10/P12/P13) against the paper-canonical Definition-B
convention (yellow-dot selection).

**Definition A — best perm by MLP fit quality (circuit R²).**
   This is what P5 records as `best_perm_idx`. Used by all downstream
   cross-topology phases. Risk: a perm can have high R² because the MLP
   fit a *flat* (uniformly-low-circuit-score) design space well — i.e.,
   "well-fit but useless." Such a perm would disagree with B and C.

**Definition B — best perm by max-of-design-space circuit_pred.**
   The argmax over `max_t circuit_pred[t]` across perms. Matches the
   DeepCirc paper yellow-dot convention (Methods L444-452): the published
   best design is whichever (perm, part-assignment) combo has the highest
   predicted circuit_score.

**Definition C — best perm by top-K-mean of circuit_pred (default K=5).**
   Robust variant of B. argmax over the mean of the top-K predictions per
   perm. Less sensitive to single-design outliers than B.

Output:
   <output_dir>/best_perm_per_topology.csv  one row per topology with:
     - topology_id, source, regulator_count, qc_tier
     - best_perm_by_r2 (A), circuit_r2_at_A, n_trained_perms
     - best_perm_by_max_circuit (B), max_circuit_at_B
     - best_perm_by_top5_mean (C, default K=5), top5_mean_at_C
     - agreement_AB, agreement_AC, agreement_ABC
     - per_perm_summary (compact JSON of (perm, r2, max_circuit, top5_mean))

   <output_dir>/best_perm_summary.json  population-level rollup:
     - n_topologies_analyzed
     - n_agree_AB, n_agree_AC, n_agree_ABC
     - rate_agree_AB (etc.)
     - per_target_disagreements (which topologies disagree, sorted by source)

**This script does NOT change downstream behavior** — it's a diagnostic.
P9/P10/P12/P13 continue to use Definition A via qc_tiers.pkl. After
inspecting the agreement rates, we decide whether to (a) keep A as the
canonical downstream choice, (b) switch to B / C, or (c) report both
in the rule book.

Path convention (Option C — group-agnostic per-topology pool; canonical
layout in scripts/_paths.py):

    `--predictions_root` is GROUP-AGNOSTIC. Post-R5 it points at the
    shared `topology_data/design_space_predictions/` pool; P7.5 reads
    .npz files BY topology_id from the manifest's loop, so it only
    accesses the ones in this group's substrate (no false-positive
    loading of other groups' .npz files even when the pool is shared).

    `--output_root` is PER-GROUP. The best_perm CSV + summary JSON are
    cumulative per-group analyses (one row per topology in this group's
    substrate); they live alongside qc_tiers.pkl etc. under
    `population/<GROUP>/`.

Usage (post-R5 group-agnostic predictions pool — recommended for G2/G3):
    python 07_5_best_perm_selection.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G2/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G2/qc_tiers.pkl \\
        --output_root         $DEEPCIRC_SCRATCH/population/G2

Usage (legacy per-group — pre-R5 G1 layout):
    python 07_5_best_perm_selection.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G1/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/population/G1/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G1/qc_tiers.pkl \\
        --output_root         $DEEPCIRC_SCRATCH/population/G1
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from _population_filter import (
    add_scope_args, filter_topologies, scope_summary,
)


def best_by_circuit_r2(perm_records: list[dict]) -> tuple[int, float]:
    """Definition A — argmax over circuit_r2."""
    best = max(perm_records, key=lambda r: r["circuit_r2"])
    return int(best["perm_idx"]), float(best["circuit_r2"])


def per_perm_design_metrics(predictions_root: Path, topology_id: str,
                             perm_idx: int) -> dict | None:
    """Read the .npz for one (topology, perm) and return summary metrics.

    Returns None if the .npz is missing.

    **v2.0 refinement (2026-05-06)**: also emits `best_design_idx` (the
    argmax-circuit position in the design space) and `best_design_assignment`
    (the part-assignment vector at that position). Used by P11 to render
    the full yellow-dot identity in the rule book — `topology_id × best_perm
    × part-assignment` as a single buildable specification.
    """
    npz_path = predictions_root / f"{topology_id}_{perm_idx}.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    circuit_pred = d["circuit_pred"].astype(np.float64)
    toxicity_pred = d["toxicity_pred"].astype(np.float64)
    gate_assignments = d["gate_assignments"]  # (N, l) int8
    n = circuit_pred.shape[0]
    if n == 0:
        return {"n_designs": 0, "max_circuit": float("-inf"),
                "top5_mean_circuit": float("-inf"),
                "p99_circuit": float("-inf"), "median_circuit": float("-inf"),
                "best_design_idx": -1, "best_design_assignment": [],
                "best_design_toxicity": float("-inf")}
    # top-K mean (sorted descending, take K), default K=5 — robust to outliers
    K = min(5, n)
    sorted_desc = np.sort(circuit_pred)[::-1]
    best_idx = int(np.argmax(circuit_pred))
    return {
        "n_designs":              int(n),
        "max_circuit":            float(circuit_pred.max()),
        "top5_mean_circuit":      float(sorted_desc[:K].mean()),
        "p99_circuit":            float(np.quantile(circuit_pred, 0.99)),
        "median_circuit":         float(np.median(circuit_pred)),
        "best_design_idx":        best_idx,
        "best_design_assignment": gate_assignments[best_idx].astype(int).tolist(),
        "best_design_toxicity":   float(toxicity_pred[best_idx]),
    }


def load_part_names(response_data_path: Path) -> list[str]:
    """20-element list of human-readable part labels: '<TF>/<RBS>'."""
    with open(response_data_path) as f:
        d = json.load(f)
    return [f"{r}/{rbs}" for r, rbs in zip(d["Repressor"], d["RBS"])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_manifest", type=Path, required=True)
    p.add_argument("--predictions_root", type=Path, required=True)
    p.add_argument("--qc_tiers", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=5,
                   help="K for top-K-mean criterion (Definition C). Default 5.")
    p.add_argument("--part_response_data", type=Path,
                   default=None,
                   help="Optional path to dgd/data/response_data_3_inputs_DeepCirc.json. "
                        "If provided, the best-design part-assignment vector is "
                        "emitted with human-readable part names for v2.0 yellow-dot "
                        "rule-book rendering.")
    add_scope_args(p)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    part_names: list[str] | None = None
    if args.part_response_data and args.part_response_data.exists():
        part_names = load_part_names(args.part_response_data)
        print(f"[setup] loaded {len(part_names)} part names from "
              f"{args.part_response_data.name}")

    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)
    with open(args.qc_tiers, "rb") as f:
        qc = pickle.load(f)
    qc_by_tid = {r["topology_id"]: r for r in qc["records"]}

    topologies = filter_topologies(
        pop["topologies"],
        include_baselines=args.include_baselines,
        size_classes=args.size_classes,
        label="P7.5 scope",
    )

    rows: list[dict] = []
    n_processed = n_skipped = 0

    for t in topologies:
        tid = t["topology_id"]
        qc_rec = qc_by_tid.get(tid)
        if qc_rec is None or qc_rec["tier"] not in ("A1", "A2"):
            n_skipped += 1
            continue

        trained = [r for r in qc_rec.get("all_perms", []) if r.get("trained")]
        if not trained:
            n_skipped += 1
            continue

        # Definition A — best by R²
        perm_A, r2_A = best_by_circuit_r2(trained)

        # Per-perm design-space metrics (load .npz files)
        per_perm: list[dict] = []
        for r in trained:
            perm_idx = int(r["perm_idx"])
            metrics = per_perm_design_metrics(args.predictions_root, tid, perm_idx)
            if metrics is None:
                # No .npz for this perm — skip (P7 may not have run for it)
                continue
            per_perm.append({
                "perm_idx":           perm_idx,
                "circuit_r2":         float(r["circuit_r2"]),
                "toxicity_r2":        float(r["toxicity_r2"]),
                **metrics,
            })

        if not per_perm:
            n_skipped += 1
            continue

        # Definition B — best by max_circuit
        best_B = max(per_perm, key=lambda x: x["max_circuit"])
        perm_B = best_B["perm_idx"]
        max_circuit_B = best_B["max_circuit"]

        # Definition C — best by top-K-mean
        best_C = max(per_perm, key=lambda x: x["top5_mean_circuit"])
        perm_C = best_C["perm_idx"]
        top5_mean_C = best_C["top5_mean_circuit"]

        # Agreement flags
        agreement_AB  = (perm_A == perm_B)
        agreement_AC  = (perm_A == perm_C)
        agreement_BC  = (perm_B == perm_C)
        agreement_ABC = (perm_A == perm_B == perm_C)

        # v2.0: yellow-dot full identity — best (perm, part-assignment) tuple
        # under Definition B (max-circuit-pred). Used by P11 rule book for
        # per-target buildable-design recommendation.
        best_design_assignment = best_B.get("best_design_assignment", [])
        best_design_toxicity   = best_B.get("best_design_toxicity",
                                            float("-inf"))
        if part_names is not None and best_design_assignment:
            best_design_part_names = [part_names[i] for i in best_design_assignment]
        else:
            best_design_part_names = None

        rows.append({
            "topology_id":              tid,
            "source":                   t.get("source"),
            "regulator_count":          t.get("regulator_count"),
            "qc_tier":                  qc_rec["tier"],
            "n_trained_perms":          len(per_perm),
            "best_perm_by_r2":          perm_A,
            "circuit_r2_at_A":          r2_A,
            "best_perm_by_max_circuit": perm_B,
            "max_circuit_at_B":         max_circuit_B,
            "best_design_toxicity_at_B": best_design_toxicity,
            "best_design_assignment":   json.dumps(best_design_assignment),
            "best_design_part_names":   (json.dumps(best_design_part_names)
                                          if best_design_part_names else ""),
            "best_perm_by_top5_mean":   perm_C,
            "top5_mean_at_C":           top5_mean_C,
            "agreement_AB":             agreement_AB,
            "agreement_AC":             agreement_AC,
            "agreement_BC":             agreement_BC,
            "agreement_ABC":            agreement_ABC,
            "per_perm_compact":         json.dumps([
                {"perm": p["perm_idx"],
                 "r2":   round(p["circuit_r2"], 4),
                 "maxc": round(p["max_circuit"], 4),
                 "topK": round(p["top5_mean_circuit"], 4)}
                for p in per_perm
            ]),
        })
        n_processed += 1

    df = pd.DataFrame(rows)
    out_csv = args.output_root / "best_perm_per_topology.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[emit] wrote {out_csv}  ({len(df)} topologies analyzed; "
          f"{n_skipped} skipped)")

    # Population summary
    n = len(df)
    if n > 0:
        n_AB  = int(df["agreement_AB"].sum())
        n_AC  = int(df["agreement_AC"].sum())
        n_BC  = int(df["agreement_BC"].sum())
        n_ABC = int(df["agreement_ABC"].sum())

        # Per-target disagreement breakdown (sorted by source for readability)
        per_target = defaultdict(lambda: {"n_total": 0, "n_disagree_AB": 0,
                                          "n_disagree_AC": 0, "n_disagree_ABC": 0})
        for _, row in df.iterrows():
            src = row["source"] if isinstance(row["source"], str) else "multi"
            per_target[src]["n_total"] += 1
            if not row["agreement_AB"]:
                per_target[src]["n_disagree_AB"] += 1
            if not row["agreement_AC"]:
                per_target[src]["n_disagree_AC"] += 1
            if not row["agreement_ABC"]:
                per_target[src]["n_disagree_ABC"] += 1

        # The interesting subset: topologies where A disagrees with B or C
        disagreements = df[~df["agreement_ABC"]].copy()
        disagreements_compact = [
            {
                "topology_id":   r["topology_id"],
                "source":        r["source"],
                "regs":          int(r["regulator_count"]),
                "perm_A_(R2)":   int(r["best_perm_by_r2"]),
                "perm_B_(maxC)": int(r["best_perm_by_max_circuit"]),
                "perm_C_(topK)": int(r["best_perm_by_top5_mean"]),
                "circuit_r2_A":  round(r["circuit_r2_at_A"], 3),
                "max_circuit_B": round(r["max_circuit_at_B"], 3),
                "top5_mean_C":   round(r["top5_mean_at_C"], 3),
            }
            for _, r in disagreements.iterrows()
        ]

        summary = {
            "scope": scope_summary(
                include_baselines=args.include_baselines,
                size_classes=args.size_classes,
            ),
            "top_k": args.top_k,
            "n_topologies_analyzed": n,
            "n_topologies_skipped":  n_skipped,
            "agreement_AB_rate":     n_AB / n,
            "agreement_AC_rate":     n_AC / n,
            "agreement_BC_rate":     n_BC / n,
            "agreement_ABC_rate":    n_ABC / n,
            "n_agree_AB":            n_AB,
            "n_agree_AC":            n_AC,
            "n_agree_BC":            n_BC,
            "n_agree_ABC":           n_ABC,
            "per_target_disagreements": dict(per_target),
            "disagreements_detail":     disagreements_compact,
        }

        out_summary = args.output_root / "best_perm_summary.json"
        with open(out_summary, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[emit] wrote {out_summary}")

        print()
        print("=" * 64)
        print(f"Phase 7.5 — best-perm selection summary (K={args.top_k})")
        print("=" * 64)
        print(f"  Topologies analyzed: {n}")
        print(f"  Agreement A==B (R² vs max_circuit):    "
              f"{n_AB}/{n} ({100*n_AB/n:.1f}%)")
        print(f"  Agreement A==C (R² vs top-{args.top_k}-mean):  "
              f"{n_AC}/{n} ({100*n_AC/n:.1f}%)")
        print(f"  Agreement B==C (max vs top-{args.top_k}-mean): "
              f"{n_BC}/{n} ({100*n_BC/n:.1f}%)")
        print(f"  Agreement A==B==C (all three):         "
              f"{n_ABC}/{n} ({100*n_ABC/n:.1f}%)")
        print()

        if n_ABC < n:
            print(f"  Disagreements (where A != B != C, by source):")
            for src, stats in sorted(per_target.items()):
                if stats["n_disagree_ABC"] > 0:
                    print(f"    {src}: {stats['n_disagree_ABC']}/{stats['n_total']}")
            print()
            print(f"  Top 10 disagreements (see best_perm_summary.json for all):")
            for d in disagreements_compact[:10]:
                # source can be str (single-target) or list (cross-target WL collision)
                src = d['source']
                src_str = ",".join(src) if isinstance(src, list) else str(src)
                print(f"    {d['topology_id']:>10s} src={src_str:<10s} "
                      f"{d['regs']}-reg  "
                      f"A={d['perm_A_(R2)']} (r²={d['circuit_r2_A']})  "
                      f"B={d['perm_B_(maxC)']} (max={d['max_circuit_B']})  "
                      f"C={d['perm_C_(topK)']} (topK={d['top5_mean_C']})")
        else:
            print("  Perfect agreement: A == B == C for every topology.")
            print("  Definition A (R²-based) is robust under this dataset.")
        print("=" * 64)
    else:
        print("[warn] 0 topologies analyzed — check qc_tiers.pkl + design_space_predictions/ paths")


if __name__ == "__main__":
    main()
