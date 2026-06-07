"""Phase 8 — L1 within-topology rule extraction.

For each (topology, perm) `.npz` produced by P7, compute three
per-design-space analyses on FOUR views of the score:

  • `circuit`      = raw `circuit_pred` (the MLP's output)
  • `circuit_log`  = `log(circuit_pred)` = m_logic per parts-only Module D
  • `toxicity`     = raw `toxicity_pred` (= paper's growth_score)
  • `toxicity_log` = `-log(toxicity_pred)` = m_burden per Module D

The raw views show what the MLP literally predicts (the substrate
downstream phases consume); the log views are in the interpretability
units used by parts-only `PROJECT_STATE.md §6.2` and are where Module D's
"circuit is pairwise, growth is slot-additive" finding lives. Both are
emitted in parallel JSON fields so downstream analyses can choose.

Per view, three analyses:

  1. **Top vs bottom 25% enrichment**
     For each (slot, part), log-odds of the part appearing in
     top-25%-by-Y vs bottom-25%-by-Y. Highlights which parts dominate
     the high end of each slot.

  2. **Single-slot variance via Hoeffding-Sobol decomposition**
     v_j = Var(E[Y | slot_j]) — variance attributable to slot j's main
     effect alone. Reported as a fraction of total Y variance.

  3. **Pairwise-slot interaction via Hoeffding-Sobol**
     v_{j1,j2} = Var(E[Y | slot_j1, slot_j2]) - v_j1 - v_j2 — pure
     pairwise interaction. Frac-of-total reported per pair and aggregated.

Per (topology, perm), one JSON written to:
    <output_root>/<topology_id>/perm_<perm_idx>.json

Per topology: 1–6 JSONs (one per trained perm). Cross-topology phases
(P9, P10) will roll up using the best-circuit-R² perm as the topology's
representative — same convention as P5.

Spot-check expectations:
  * Raw circuit higher_order is target-dependent — pairwise-clean for
    structurally-simple targets like 0xEE (~0.2), 3+-body-rich for
    asymmetric targets like 0x17 / 0x2B / 0x6D (~0.6+). Real finding,
    not artifact.
  * Log-circuit (m_logic) on paper-anchor topologies should reproduce
    Module D's "circuit is structurally pairwise" — main+pair ≥ 0.7,
    higher_order ≤ 0.3.
  * Raw + log toxicity should both be slot-additive-dominated — main
    ≥ 0.5, pair ≤ 0.3, higher_order growing slowly with gate_count.

Path convention (Option C — group-agnostic per-topology pool; canonical
layout in scripts/_paths.py):

    Both `--predictions_root` and `--output_root` are GROUP-AGNOSTIC
    under post-R5. Post-R5 they point at the shared per-topology pool:
    `topology_data/design_space_predictions/` (P7's output) and
    `topology_data/l1/` (P8's output). G2's P8 reuses G1's existing
    per-(topology, perm) JSONs via the file-level idempotency check
    (out_json.exists()) — only NEW topologies in G2's substrate get
    L1 records computed.

    `--population_manifest` and `--qc_tiers` remain per-group inputs
    (they describe *which* topologies to analyze for this group's
    cumulative L1 record set).

Usage (post-R5 group-agnostic — recommended for G2/G3 fires):
    python 08_l1_within_topology.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G2/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --output_root         $DEEPCIRC_SCRATCH/topology_data/l1 \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G2/qc_tiers.pkl

Usage (legacy per-group — pre-R5 G1 layout):
    python 08_l1_within_topology.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G1/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/population/G1/design_space_predictions \\
        --output_root         $DEEPCIRC_SCRATCH/population/G1/l1 \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G1/qc_tiers.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from _population_filter import add_scope_args, filter_topologies

NUM_PARTS = 20
ZERO_PERMS_MARKER = "_zero_perms_filtered.json"


def top_bottom_enrichment(gate_assignments: np.ndarray, Y: np.ndarray,
                          quantile: float = 0.25
                          ) -> tuple[float, float, np.ndarray]:
    """For each (slot, part), log-odds of the part in top-quantile vs bottom-quantile.

    gate_assignments: (N, l) int part indices
    Y:                (N,) prediction values
    quantile:         tail width (0.25 = top/bottom 25%)

    Returns (top_threshold, bot_threshold, log_odds[l, 20]).
    Laplace smoothing (+1) keeps log-odds finite when a part is absent in
    one tail.
    """
    n, l = gate_assignments.shape
    cut_top = float(np.quantile(Y, 1.0 - quantile))
    cut_bot = float(np.quantile(Y, quantile))
    top_mask = Y >= cut_top
    bot_mask = Y <= cut_bot

    log_odds = np.zeros((l, NUM_PARTS), dtype=np.float64)
    for j in range(l):
        parts_j = gate_assignments[:, j].astype(np.int64)
        top_count = np.bincount(parts_j[top_mask], minlength=NUM_PARTS).astype(np.float64)
        bot_count = np.bincount(parts_j[bot_mask], minlength=NUM_PARTS).astype(np.float64)
        log_odds[j] = np.log((top_count + 1.0) / (bot_count + 1.0))
    return cut_top, cut_bot, log_odds


def conditional_means_for_slot(parts_j: np.ndarray, Y: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
    """E[Y | slot_j = p] per part p, plus the per-design conditional mean.

    Returns (mean_per_part[NUM_PARTS], cond_mean[N]).
    Empty parts (count=0) get mean 0; they don't contribute to variance
    decomposition because their mass in cond_mean is also 0.
    """
    count = np.bincount(parts_j, minlength=NUM_PARTS).astype(np.float64)
    weighted = np.bincount(parts_j, weights=Y, minlength=NUM_PARTS).astype(np.float64)
    mean_per_part = np.where(count > 0, weighted / np.maximum(count, 1.0), 0.0)
    cond_mean = mean_per_part[parts_j]
    return mean_per_part, cond_mean


def conditional_means_for_pair(parts_j1: np.ndarray, parts_j2: np.ndarray,
                                Y: np.ndarray) -> np.ndarray:
    """E[Y | slot_j1=p1, slot_j2=p2] expanded to per-design (N,).

    Joint cell index = parts_j1 * NUM_PARTS + parts_j2.
    """
    joint = parts_j1.astype(np.int64) * NUM_PARTS + parts_j2.astype(np.int64)
    n_cells = NUM_PARTS * NUM_PARTS
    count = np.bincount(joint, minlength=n_cells).astype(np.float64)
    weighted = np.bincount(joint, weights=Y, minlength=n_cells).astype(np.float64)
    mean_per_cell = np.where(count > 0, weighted / np.maximum(count, 1.0), 0.0)
    return mean_per_cell[joint]


def hoeffding_sobol_decomposition(gate_assignments: np.ndarray,
                                   Y: np.ndarray) -> dict:
    """Variance decomposition: v_j (main) and v_{j1,j2} (pairwise interaction).

    Total Y variance = sum_main + sum_pairwise + higher_order_residual.
    """
    n, l = gate_assignments.shape
    Y_mean = float(Y.mean())
    Y_var = float(Y.var())

    # Main effects: v_j = Var(E[Y | slot_j])
    cond_mean_per_slot = np.zeros((l, n), dtype=np.float64)
    v_main = np.zeros(l, dtype=np.float64)
    for j in range(l):
        parts_j = gate_assignments[:, j].astype(np.int64)
        _, cond = conditional_means_for_slot(parts_j, Y)
        cond_mean_per_slot[j] = cond
        # Var(E[Y | slot_j]) = E[(E[Y|j] - E[Y])²]
        v_main[j] = float(((cond - Y_mean) ** 2).mean())

    # Pairwise: v_{j1,j2} = Var(E[Y | slot_j1, slot_j2]) - v_j1 - v_j2
    pairwise: dict[tuple[int, int], float] = {}
    for j1, j2 in combinations(range(l), 2):
        cond_jj = conditional_means_for_pair(
            gate_assignments[:, j1].astype(np.int64),
            gate_assignments[:, j2].astype(np.int64),
            Y,
        )
        v_jj_total = float(((cond_jj - Y_mean) ** 2).mean())
        pairwise[(j1, j2)] = max(v_jj_total - v_main[j1] - v_main[j2], 0.0)

    sum_main = float(v_main.sum())
    sum_pair = float(sum(pairwise.values()))
    residual = max(Y_var - sum_main - sum_pair, 0.0)

    return {
        "Y_mean":              Y_mean,
        "Y_var":               Y_var,
        "main_var":            v_main.tolist(),
        "main_var_frac":       (v_main / max(Y_var, 1e-12)).tolist(),
        "main_total_frac":     sum_main / max(Y_var, 1e-12),
        "pairwise_var":        {f"{j1},{j2}": v for (j1, j2), v in pairwise.items()},
        "pairwise_var_frac":   {f"{j1},{j2}": v / max(Y_var, 1e-12)
                                 for (j1, j2), v in pairwise.items()},
        "pairwise_total_frac": sum_pair / max(Y_var, 1e-12),
        "higher_order_frac":   residual / max(Y_var, 1e-12),
    }


def analyze_one_npz(npz_path: Path, qc_tier: str | None) -> dict:
    """Load an .npz, run three analyses on circuit + toxicity, return a record."""
    d = np.load(npz_path, allow_pickle=True)
    gate_assignments = d["gate_assignments"]   # (N, l) int8
    circuit_pred = d["circuit_pred"]            # (N,) float32
    toxicity_pred = d["toxicity_pred"]          # (N,) float32
    meta = d["meta"].item()                     # dict

    n, l = gate_assignments.shape
    record: dict = {
        "topology_id": meta["topology_id"],
        "perm_idx":    meta["perm_idx"],
        "qc_tier":     qc_tier or meta.get("qc_tier"),
        "meta":        {k: meta[k] for k in
                        ("topology_id", "nngga_circuit_name", "gate_count",
                         "perm_idx", "source", "energy", "is_baseline",
                         "is_parser_pick") if k in meta},
        "n_designs":   int(n),
        "n_slots":     int(l),
    }

    # Raw + log-transformed views per Module D's interpretability convention:
    #   m_logic  = log(circuit_score)    per parts-only PROJECT_STATE.md §6.2
    #   m_burden = -log(growth_score)    per same source
    # Raw circuit has a fat right tail (range often 0.2-100x); raw growth
    # is bounded in [0, 1]. Log-transforms compress tails to surface the
    # pairwise structure Module D identified ("circuit is pairwise, growth
    # is slot-additive"). Clip to FLOOR to avoid log(0).
    FLOOR = 1e-6
    circuit_pred_f = circuit_pred.astype(np.float64)
    toxicity_pred_f = toxicity_pred.astype(np.float64)
    tasks: list[tuple[str, np.ndarray]] = [
        ("circuit",      circuit_pred_f),
        ("circuit_log",  np.log(np.maximum(circuit_pred_f, FLOOR))),
        ("toxicity",     toxicity_pred_f),
        ("toxicity_log", -np.log(np.maximum(toxicity_pred_f, FLOOR))),  # m_burden
    ]
    for task, Y in tasks:
        cut_top, cut_bot, log_odds = top_bottom_enrichment(gate_assignments, Y)
        sobol = hoeffding_sobol_decomposition(gate_assignments, Y)
        record[task] = {
            "pred_min":              float(Y.min()),
            "pred_max":              float(Y.max()),
            "pred_mean":             float(Y.mean()),
            "pred_median":           float(np.median(Y)),
            "pred_std":              float(Y.std()),
            "top_quantile_threshold": cut_top,
            "bot_quantile_threshold": cut_bot,
            "enrichment_top_vs_bot": log_odds.tolist(),
            **sobol,
        }
    return record


def write_record_json(record: dict, output_root: Path) -> Path:
    """Write a per-(topology, perm) record to <output_root>/<tid>/<perm>.json."""
    tid = record["topology_id"]
    perm_idx = record["perm_idx"]
    target_dir = output_root / tid
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"perm_{perm_idx}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=float)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_manifest", type=Path, required=True)
    p.add_argument("--predictions_root", type=Path, required=True,
                   help="Root holding the .npz files from P7")
    p.add_argument("--output_root", type=Path, required=True,
                   help="Per-topology JSONs land at <output_root>/<topology_id>/")
    p.add_argument("--qc_tiers", type=Path, default=None,
                   help="Optional qc_tiers.pkl from P5 — used to skip A3/A0/BROKEN")
    p.add_argument("--limit_topologies", type=int, default=None,
                   help="If set, process only the first N topologies (for testing)")
    add_scope_args(p)
    args = p.parse_args()

    if not args.population_manifest.exists():
        raise SystemExit(f"missing population manifest: {args.population_manifest}")
    if not args.predictions_root.is_dir():
        raise SystemExit(f"missing predictions_root: {args.predictions_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)

    skip_ids: set[str] = set()
    tier_for_topology: dict[str, str] = {}
    if args.qc_tiers and args.qc_tiers.exists():
        with open(args.qc_tiers, "rb") as f:
            qc = pickle.load(f)
        for r in qc["records"]:
            tier_for_topology[r["topology_id"]] = r["tier"]
            if r["tier"] in ("A3", "A0", "BROKEN"):
                skip_ids.add(r["topology_id"])
        print(f"[qc] loaded {args.qc_tiers.name}; skipping "
              f"{len(skip_ids)} topologies tagged A3/A0/BROKEN")

    topologies = filter_topologies(
        pop["topologies"],
        include_baselines=args.include_baselines,
        size_classes=args.size_classes,
        label="P8 scope",
    )
    if args.limit_topologies is not None:
        topologies = topologies[:args.limit_topologies]
        print(f"[limit] processing first {len(topologies)} topologies")

    n_topologies_done = n_skipped = n_perms_done = n_already = n_error = 0
    t0 = time.perf_counter()

    for i, t in enumerate(topologies, start=1):
        tid = t["topology_id"]
        tier = tier_for_topology.get(tid, "?")
        if tid in skip_ids:
            print(f"[{i}/{len(topologies)}] {tid} ({t['regulator_count']}-reg) "
                  f"-- SKIP (tier={tier})")
            n_skipped += 1
            continue

        # Find all per-perm .npz files for this topology
        npz_files = sorted(args.predictions_root.glob(f"{tid}_*.npz"))
        if not npz_files:
            print(f"[{i}/{len(topologies)}] {tid} ({t['regulator_count']}-reg, "
                  f"tier={tier}) -- no .npz files found, skipping")
            n_skipped += 1
            continue

        print(f"[{i}/{len(topologies)}] {tid} ({t['regulator_count']}-reg, "
              f"src={t['source']}, tier={tier}, npz={len(npz_files)})")

        for npz_path in npz_files:
            perm_idx = int(npz_path.stem.split("_")[-1])
            out_json = args.output_root / tid / f"perm_{perm_idx}.json"
            if out_json.exists():
                print(f"  perm {perm_idx:>2}: already_done")
                n_already += 1
                continue
            try:
                t_start = time.perf_counter()
                record = analyze_one_npz(npz_path, qc_tier=tier)
                write_record_json(record, args.output_root)
                elapsed = time.perf_counter() - t_start
                print(f"  perm {perm_idx:>2}: {record['n_designs']:>10,} designs  "
                      f"in {elapsed:.2f}s")
                n_perms_done += 1
            except Exception as e:
                print(f"  perm {perm_idx:>2}: ERROR {type(e).__name__}: {e}")
                n_error += 1
        n_topologies_done += 1

    print()
    print(f"=== Phase 8 done in {(time.perf_counter() - t0)/60:.1f} min ===")
    print(f"  topologies processed: {n_topologies_done}")
    print(f"  skipped (A0/A3/no_npz): {n_skipped}")
    print(f"  perm-records written: {n_perms_done}")
    print(f"  perm-records already done: {n_already}")
    print(f"  errors: {n_error}")
    print(f"  output dir: {args.output_root}")


if __name__ == "__main__":
    main()
