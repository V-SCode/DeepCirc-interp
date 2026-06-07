"""Phase 9 — L2 cross-topology graph-role rule extraction.

For each topology in tier A1+A2, take the **best-circuit-R² perm** (the P5
representative) and pool its `circuit_log` + `toxicity` design-space
predictions across the population, indexed by **per-slot graph-role
fingerprint** (computed in P3 §7.1) instead of slot index.

Goal: find (graph_role, part) cells where the part is enriched in the
top-25%-by-pred designs across multiple topologies. These are the
cross-topology design rules that generalize beyond a single topology
and are the substrate for the L4 rule book.

Per task (`circuit_log`, `toxicity`):
  - For each (fingerprint_key, part_idx) cell:
      • count of top-25%-by-pred designs containing that part at a slot
        with that fingerprint
      • count of bot-25%-by-pred designs likewise
      • log-odds = log((top + 1) / (bot + 1))
      • num_topologies and num_targets contributing
      • per-target breakdown (so target-specific vs target-agnostic
        rules can be distinguished)

The cell key is the **full** fingerprint — option (a) per
PROJECT_STATE_TOPOLOGY.md §1.2 design choice — concatenating
(depth_to_output, depth_from_input, in_degree, out_degree,
predecessor_role_signature, successor_role_signature, node_type)
into a single canonical string. This means only slots with *exactly
identical* role profiles get pooled; finer-grained but more cells.

Output:
  <output_root>/l2_enrichment.csv          one row per (fp_key, part, task)
  <output_root>/l2_enrichment_by_target.csv  one row per (fp_key, part, task, source)
  <output_root>/l2_summary.json            top-K rules + meta

(CSV chosen over parquet for portability; the data is small enough — ~7k
aggregate + ~30k per-target rows — that compression isn't important.)

Compute: per-topology pass is np.bincount-style, ~1s each. ~5-10 min total.

Path convention (Option C — group-agnostic per-topology pool; canonical
layout in scripts/_paths.py):

    `--predictions_root` is GROUP-AGNOSTIC. Post-R5 it points at the
    shared `topology_data/design_space_predictions/` pool; P9 reads
    .npz files BY topology_id from the manifest's loop, so it only
    accesses .npz files for topologies in this group's substrate (no
    false-positive loading of other groups' files even from a shared pool).

    `--output_root` is PER-GROUP. The L2 enrichment CSVs (universal +
    per-target) are cumulative per-group analyses; they live under
    `population/<GROUP>/l2/`.

Tier cutoffs are parametric via `--top_quantile` / `--bot_quantile`
(defaults 0.75 / 0.25 — i.e., top-25% vs bot-25%, the v2.0 rule-book
default). To compute the narrower top-5% / bot-5% enrichment used by
figS05 / figS06, pass `--top_quantile 0.95 --bot_quantile 0.05` and
direct `--output_root` to a sibling dir (e.g. `.../l2_top05/`) so the
canonical 25/25 outputs are preserved.

Usage (post-R5 group-agnostic — recommended for G2/G3 fires):
    python 09_l2_graph_role.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G2/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G2/qc_tiers.pkl \\
        --output_root         $DEEPCIRC_SCRATCH/population/G2/l2 \\
        --part_response_data  $DEEPCIRC_BASE/DeepCirc_upstream/dgd/data/response_data_3_inputs_DeepCirc.json

Usage (legacy per-group — pre-R5 G1 layout):
    python 09_l2_graph_role.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G1/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/population/G1/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G1/qc_tiers.pkl \\
        --output_root         $DEEPCIRC_SCRATCH/population/G1/l2 \\
        --part_response_data  $DEEPCIRC_BASE/DeepCirc_upstream/dgd/data/response_data_3_inputs_DeepCirc.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from _population_filter import (
    add_scope_args, filter_topologies, scope_summary,
)

NUM_PARTS = 20
ZERO_PERMS_MARKER = "_zero_perms_filtered.json"
FLOOR = 1e-6


def load_part_names(response_data_path: Path) -> list[str]:
    """Read the upstream response_data_3_inputs JSON and emit a 20-element
    list of human-readable part labels: '<Repressor>/<RBS>' (e.g. 'IcaRA/I1').
    """
    with open(response_data_path) as f:
        d = json.load(f)
    repressors = d["Repressor"]
    rbss = d["RBS"]
    if len(repressors) != NUM_PARTS or len(rbss) != NUM_PARTS:
        raise SystemExit(
            f"part library size mismatch: expected {NUM_PARTS} parts but "
            f"response_data has {len(repressors)} repressors / {len(rbss)} RBSs"
        )
    return [f"{r}/{rbs}" for r, rbs in zip(repressors, rbss)]


def fingerprint_key(fp: dict) -> str:
    """Canonical role-signature key for one slot.

    Uses only the role-defining fields (dropping `node` (graph node id),
    topology_id, target_hex, regulator_count, slot_index — those are
    bookkeeping). Joined into a fixed-order pipe-separated string so
    equality is well-defined.

    Field naming follows P3's emit format
    (`topology/scripts/03_assemble_population.py`):
      • `predecessor_depth_signature` — sorted depths of immediate
        predecessors, e.g. "0,2"
      • Successor signature is not currently emitted by P3 (state doc
        §7.1 specced it but the implementation only does predecessors).
        Added a `succ=NA` placeholder so the key shape is forward-
        compatible if P3 is later extended.
    """
    return "|".join([
        f"d2o={fp.get('depth_to_output', 'NA')}",
        f"dfi={fp.get('depth_from_input', 'NA')}",
        f"in={fp.get('in_degree', 'NA')}",
        f"out={fp.get('out_degree', 'NA')}",
        f"pred={fp.get('predecessor_depth_signature', 'NA')}",
        f"succ={fp.get('successor_depth_signature', 'NA')}",
        f"type={fp.get('node_type', 'NA')}",
    ])


def best_perm_for_topology(qc_record: dict) -> int | None:
    """Return the perm_idx with highest circuit R² (P5's anchor convention).

    Returns None if topology has no trained perms (shouldn't happen for
    A1/A2 records — caller should already have filtered out A0/A3/BROKEN).
    """
    if "best_perm_idx" in qc_record:
        return qc_record["best_perm_idx"]
    perms = [p for p in qc_record.get("all_perms", []) if p.get("trained")]
    if not perms:
        return None
    return max(perms, key=lambda p: p["circuit_r2"])["perm_idx"]


def aggregate_one_topology(
    npz_path: Path,
    topology_meta: dict,
    fingerprints: list[dict],
    counters: dict,
    counters_by_target: dict,
    top_quantile: float = 0.75,
    bot_quantile: float = 0.25,
) -> dict:
    """Stream one (topology, perm) .npz into the global counters.

    Updates `counters[task][fp_key][part]` += count for both top and bot
    tier (per task = circuit_log / toxicity). Tier cutoffs default to
    Q3 / Q1 (top-25% / bot-25%); pass narrower values (e.g. 0.95 / 0.05)
    to compute top-5% / bot-5% enrichment.
    `counters_by_target[task][source][fp_key][part]` similarly.
    Returns per-topology stats for the run summary.
    """
    d = np.load(npz_path, allow_pickle=True)
    gate_assignments = d["gate_assignments"]            # (N, l) int8
    circuit_pred = d["circuit_pred"].astype(np.float64)  # (N,)
    toxicity_pred = d["toxicity_pred"].astype(np.float64)
    n, l = gate_assignments.shape
    assert l == len(fingerprints), (
        f"slot count mismatch: gate_assignments has {l} slots, "
        f"fingerprints has {len(fingerprints)}"
    )

    raw_source = topology_meta.get("source", "unknown")
    # Multi-source topologies (cross-target WL collisions) have source as list;
    # iterate to attribute counters to each contributing target.
    if isinstance(raw_source, list):
        sources = raw_source if raw_source else ["unknown"]
    else:
        sources = [raw_source] if raw_source else ["unknown"]
    fp_keys = [fingerprint_key(fp) for fp in fingerprints]

    # m_logic — log-transformed circuit, per Module D convention
    Y_per_task = {
        "circuit_log": np.log(np.maximum(circuit_pred, FLOOR)),
        "toxicity":    toxicity_pred,
    }

    n_topology_top_designs = 0
    n_topology_bot_designs = 0

    for task, Y in Y_per_task.items():
        top_thr = float(np.quantile(Y, top_quantile))
        bot_thr = float(np.quantile(Y, bot_quantile))
        top_mask = Y >= top_thr
        bot_mask = Y <= bot_thr
        if task == "circuit_log":
            n_topology_top_designs = int(top_mask.sum())
            n_topology_bot_designs = int(bot_mask.sum())

        for j in range(l):
            parts_j = gate_assignments[:, j].astype(np.int64)
            top_counts = np.bincount(parts_j[top_mask], minlength=NUM_PARTS)
            bot_counts = np.bincount(parts_j[bot_mask], minlength=NUM_PARTS)
            fk = fp_keys[j]

            # Population-aggregate counters
            counters[task][fk]["top"] += top_counts
            counters[task][fk]["bot"] += bot_counts
            counters[task][fk]["topologies"].add(topology_meta["topology_id"])
            for src in sources:
                counters[task][fk]["targets"].add(src)
                # Per-target counters: attribute this topology's data to each
                # source target it belongs to (multi-source topologies count
                # in each contributing target's per-target table).
                counters_by_target[task][src][fk]["top"] += top_counts
                counters_by_target[task][src][fk]["bot"] += bot_counts

    return {
        "topology_id":     topology_meta["topology_id"],
        "source":          raw_source,
        "n_designs":       n,
        "n_slots":         l,
        "n_top_designs":   n_topology_top_designs,
        "n_bot_designs":   n_topology_bot_designs,
        "fingerprint_keys": fp_keys,
    }


def cell_factory():
    return {
        "top": np.zeros(NUM_PARTS, dtype=np.int64),
        "bot": np.zeros(NUM_PARTS, dtype=np.int64),
        "topologies": set(),
        "targets":    set(),
    }


def cell_factory_target():
    return {
        "top": np.zeros(NUM_PARTS, dtype=np.int64),
        "bot": np.zeros(NUM_PARTS, dtype=np.int64),
    }


def emit_aggregate_dataframe(counters: dict, part_names: list[str]) -> pd.DataFrame:
    """Flatten counters dict into a long-format DataFrame.

    Columns:
        task, fp_key, depth_to_output, ..., node_type,   <- fingerprint fields
        part_idx, part_name,
        top_count, bot_count, log_odds,
        n_topologies, n_targets, total_designs
    """
    rows = []
    # log_odds is emitted in base-2 (log₂) for cross-figure consistency with
    # the L3 motif pipeline (P26 / P27 / fig30 all use log₂). Computed as
    # log2(x) = log(x) / log(2) to match P26's idiom (26_l3_motif_size_aggregation.py).
    LOG2 = np.log(2.0)
    for task, fp_dict in counters.items():
        for fk, cell in fp_dict.items():
            top = cell["top"]
            bot = cell["bot"]
            log_odds = np.log((top.astype(np.float64) + 1.0) / (bot.astype(np.float64) + 1.0)) / LOG2
            n_topo = len(cell["topologies"])
            n_targ = len(cell["targets"])
            for part_idx in range(NUM_PARTS):
                rows.append({
                    "task":      task,
                    "fp_key":    fk,
                    **dict(parse_fp_key(fk)),
                    "part_idx":  part_idx,
                    "part_name": part_names[part_idx],
                    "top_count": int(top[part_idx]),
                    "bot_count": int(bot[part_idx]),
                    "log_odds":  float(log_odds[part_idx]),
                    "n_topologies": n_topo,
                    "n_targets":    n_targ,
                    "total_designs": int(top[part_idx] + bot[part_idx]),
                })
    return pd.DataFrame(rows)


def emit_per_target_dataframe(counters_by_target: dict, part_names: list[str]) -> pd.DataFrame:
    """Same as emit_aggregate_dataframe but with `source` column."""
    rows = []
    # Same log₂ convention as emit_aggregate_dataframe — keep in sync.
    LOG2 = np.log(2.0)
    for task, target_dict in counters_by_target.items():
        for source, fp_dict in target_dict.items():
            for fk, cell in fp_dict.items():
                top = cell["top"]
                bot = cell["bot"]
                log_odds = np.log((top.astype(np.float64) + 1.0) / (bot.astype(np.float64) + 1.0)) / LOG2
                for part_idx in range(NUM_PARTS):
                    rows.append({
                        "task":      task,
                        "source":    source,
                        "fp_key":    fk,
                        **dict(parse_fp_key(fk)),
                        "part_idx":  part_idx,
                        "part_name": part_names[part_idx],
                        "top_count": int(top[part_idx]),
                        "bot_count": int(bot[part_idx]),
                        "log_odds":  float(log_odds[part_idx]),
                        "total_designs": int(top[part_idx] + bot[part_idx]),
                    })
    return pd.DataFrame(rows)


def parse_fp_key(fk: str) -> list[tuple[str, str]]:
    """Decompose `d2o=1|dfi=2|in=...` back into individual columns.
    Returns list of (column_name, value_str) so it can be spread into a row.
    """
    out = []
    for piece in fk.split("|"):
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        col_map = {
            "d2o":  "depth_to_output",
            "dfi":  "depth_from_input",
            "in":   "in_degree",
            "out":  "out_degree",
            "pred": "predecessor_depth_signature",
            "succ": "successor_depth_signature",
            "type": "node_type",
        }
        out.append((col_map.get(k, k), v))
    return out


def emit_top_rules_summary(
    df: pd.DataFrame, top_k: int = 30, min_topologies: int = 2,
    min_total_designs: int = 1000,
) -> dict:
    """Pick the strongest cross-topology rules per task.

    A rule is a (fp_key, part) cell with:
      - log_odds magnitude high
      - present in >= `min_topologies` topologies (so it's not single-topology)
      - total_designs >= `min_total_designs` (so finite-sample noise is small)

    Returns a dict keyed by task with top_k rules each.
    """
    summary: dict = {}
    for task in df["task"].unique():
        d = df[(df["task"] == task)
                & (df["n_topologies"] >= min_topologies)
                & (df["total_designs"] >= min_total_designs)
                ].copy()
        d["abs_log_odds"] = d["log_odds"].abs()
        top = d.nlargest(top_k, "abs_log_odds")
        summary[task] = []
        for _, row in top.iterrows():
            summary[task].append({
                "part_name":     row["part_name"],
                "part_idx":      int(row["part_idx"]),
                "log_odds":      float(row["log_odds"]),
                "direction":     "enriched in top" if row["log_odds"] > 0 else "depleted in top",
                "fp_key":        row["fp_key"],
                "n_topologies":  int(row["n_topologies"]),
                "n_targets":     int(row["n_targets"]),
                "top_count":     int(row["top_count"]),
                "bot_count":     int(row["bot_count"]),
            })
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_manifest", type=Path, required=True)
    p.add_argument("--predictions_root", type=Path, required=True)
    p.add_argument("--qc_tiers", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--part_response_data", type=Path, required=True,
                   help="Path to dgd/data/response_data_3_inputs_DeepCirc.json")
    p.add_argument("--top_quantile", type=float, default=0.75,
                   help="Per-topology quantile cutoff for the top tier "
                        "(default 0.75 = top-25%%; pass 0.95 for top-5%%)")
    p.add_argument("--bot_quantile", type=float, default=0.25,
                   help="Per-topology quantile cutoff for the bot tier "
                        "(default 0.25 = bot-25%%; pass 0.05 for bot-5%%)")
    p.add_argument("--limit_topologies", type=int, default=None)
    add_scope_args(p)
    args = p.parse_args()

    if not 0.0 < args.bot_quantile < args.top_quantile < 1.0:
        raise SystemExit(
            f"bad tier cutoffs: require 0 < bot_quantile ({args.bot_quantile}) "
            f"< top_quantile ({args.top_quantile}) < 1"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Load population manifest + qc_tiers
    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)
    with open(args.qc_tiers, "rb") as f:
        qc = pickle.load(f)
    qc_by_tid = {r["topology_id"]: r for r in qc["records"]}

    part_names = load_part_names(args.part_response_data)
    print(f"[setup] {len(part_names)} parts loaded; "
          f"first three: {part_names[:3]}; "
          f"part 7 (Module-A growth-toxic): {part_names[7]}")

    # Initialize global counters
    counters = {
        "circuit_log": defaultdict(cell_factory),
        "toxicity":    defaultdict(cell_factory),
    }
    counters_by_target = {
        "circuit_log": defaultdict(lambda: defaultdict(cell_factory_target)),
        "toxicity":    defaultdict(lambda: defaultdict(cell_factory_target)),
    }

    # Per-topology pass — filter to active analysis scope first
    topologies = filter_topologies(
        pop["topologies"],
        include_baselines=args.include_baselines,
        size_classes=args.size_classes,
        label="P9 scope",
    )
    if args.limit_topologies is not None:
        topologies = topologies[:args.limit_topologies]

    n_processed = n_skipped = 0
    per_topology_log: list[dict] = []
    t0 = time.perf_counter()

    for i, t in enumerate(topologies, start=1):
        tid = t["topology_id"]
        qc_rec = qc_by_tid.get(tid)
        if qc_rec is None:
            print(f"[{i}/{len(topologies)}] {tid} -- no QC record, skipping")
            n_skipped += 1
            continue
        if qc_rec["tier"] in ("A3", "A0", "BROKEN"):
            print(f"[{i}/{len(topologies)}] {tid} -- SKIP (tier={qc_rec['tier']})")
            n_skipped += 1
            continue

        best_perm = best_perm_for_topology(qc_rec)
        if best_perm is None:
            print(f"[{i}/{len(topologies)}] {tid} -- no best perm found, skipping")
            n_skipped += 1
            continue

        npz_path = args.predictions_root / f"{tid}_{best_perm}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(topologies)}] {tid} -- no .npz for best perm "
                  f"{best_perm}, skipping")
            n_skipped += 1
            continue

        fingerprints = t.get("fingerprints")
        if fingerprints is None:
            print(f"[{i}/{len(topologies)}] {tid} -- no fingerprints in manifest, "
                  f"skipping (P3 may need re-running)")
            n_skipped += 1
            continue

        t_start = time.perf_counter()
        try:
            stats = aggregate_one_topology(npz_path, t, fingerprints,
                                            counters, counters_by_target,
                                            top_quantile=args.top_quantile,
                                            bot_quantile=args.bot_quantile)
            elapsed = time.perf_counter() - t_start
            per_topology_log.append({**stats, "elapsed_s": elapsed,
                                      "best_perm": best_perm,
                                      "tier": qc_rec["tier"]})
            print(f"[{i}/{len(topologies)}] {tid} ({t['regulator_count']}-reg, "
                  f"src={t['source']}, perm={best_perm}, tier={qc_rec['tier']}) "
                  f"-> {stats['n_designs']:>10,} designs in {elapsed:.1f}s")
            n_processed += 1
        except Exception as e:
            print(f"[{i}/{len(topologies)}] {tid} -- ERROR {type(e).__name__}: {e}")
            n_skipped += 1

    print()
    print(f"[aggregate] processed {n_processed} topologies, skipped {n_skipped}")
    print(f"[aggregate] {sum(len(c) for c in counters.values())} (task, fp_key) cells")

    # Emit DataFrames
    print(f"[emit] flattening counters -> DataFrames")
    df_agg = emit_aggregate_dataframe(counters, part_names)
    df_per_target = emit_per_target_dataframe(counters_by_target, part_names)

    out_agg = args.output_root / "l2_enrichment.csv"
    out_per_target = args.output_root / "l2_enrichment_by_target.csv"
    df_agg.to_csv(out_agg, index=False)
    df_per_target.to_csv(out_per_target, index=False)
    print(f"[emit] wrote {out_agg}  ({len(df_agg):,} rows)")
    print(f"[emit] wrote {out_per_target}  ({len(df_per_target):,} rows)")

    # Top-K rules summary
    print(f"[emit] top-30 cross-topology rules per task")
    summary = emit_top_rules_summary(df_agg, top_k=30, min_topologies=2,
                                      min_total_designs=1000)
    out_summary = args.output_root / "l2_summary.json"
    with open(out_summary, "w") as f:
        json.dump({
            "scope": scope_summary(
                include_baselines=args.include_baselines,
                size_classes=args.size_classes,
            ),
            "n_topologies_processed": n_processed,
            "n_topologies_skipped":   n_skipped,
            "n_unique_fp_keys":       sum(len(c) for c in counters.values()),
            "thresholds": {
                "min_topologies":    2,
                "min_total_designs": 1000,
                "top_k_per_task":    30,
            },
            "tier_cutoffs": {
                "top_quantile": args.top_quantile,
                "bot_quantile": args.bot_quantile,
            },
            "top_rules_per_task": summary,
            "per_topology":       per_topology_log,
        }, f, indent=2)
    print(f"[emit] wrote {out_summary}")

    # Console preview
    print()
    print("=== Top-10 cross-topology rules per task (preview) ===")
    for task, rules in summary.items():
        print(f"\n  {task}:")
        for j, r in enumerate(rules[:10]):
            print(f"    {j+1:2d}. {r['part_name']:>10s}  "
                  f"log_odds={r['log_odds']:+.2f}  "
                  f"({r['direction']}, n_topo={r['n_topologies']}, "
                  f"n_targ={r['n_targets']})")
            print(f"          @ {r['fp_key']}")

    print()
    print(f"=== Phase 9 done in {(time.perf_counter() - t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
