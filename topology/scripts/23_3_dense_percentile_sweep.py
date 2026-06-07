"""Phase 23.3 — Dense per-size percentile sweep for fig31_v3.

Variant of Phase 23.2 (`23_2_size_tier_design_counts.py`) that reuses
the same per-size histogram aggregation but emits a DENSE percentile
sweep (~1000 anchors) instead of 8. Designed for fig31_v3's
"empirical-CDF-inverse" curves where every point on the curve is a
direct percentile readout from the size-stratified histogram.

Method: identical to P23.2 step 1+2 (single-pass per-topology 200k-bin
histograms, aggregated per size class). Step 3 (per-topology tier
counts) is REPLACED by a dense percentile readout. No per-topology CSV
emitted — fig31_v3 only needs the per-size percentile vectors.

Anchor list (1003 points per (size × task)):

    [0.01, 0.05]                       — bottom-tail detail
    [0.1, 0.2, ..., 99.8, 99.9]        — 999 evenly-spaced anchors
    [99.95, 99.99]                     — top-tail cliff detail

p50 is included naturally (one of the 999 step-0.1 anchors).

Output (under --output_root):

    dense_percentile_sweep.json
        anchors_pct: [...1003 floats...]
        per_size:
            "5": {
                "circuit_log": [...1003 floats, one per anchor...],
                "toxicity":    [...1003 floats...],
                "n_designs_total_circuit": int,
                "n_designs_total_toxicity": int,
                "n_topologies": int,
            }
            "6": ...
            "7": ...

Plus: scope, hist_resolution, elapsed_s, phase metadata.

Compute: ~10-15 min CPU wall total for 215 topologies (same as P23.2 —
single .npz pass; the dense readout step is microseconds since it just
walks an in-memory cumulative-sum array).

Usage on cluster:

    python topology/scripts/23_3_dense_percentile_sweep.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G3/population.pkl \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G3/qc_tiers.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --output_root         $DEEPCIRC_SCRATCH/population/G3/size_tier_designs

Then transfer + push from cluster (see header of 23_2 for the standard
git workflow).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent

# Reuse best_perm_for_topology from P9
_spec = importlib.util.spec_from_file_location("_p9", _HERE / "09_l2_graph_role.py")
_p9 = importlib.util.module_from_spec(_spec)
sys.modules["_p9"] = _p9
_spec.loader.exec_module(_p9)
best_perm_for_topology = _p9.best_perm_for_topology

# Reuse histogram helpers from P23 (numeric-prefix scripts can't be import'd normally)
_spec2 = importlib.util.spec_from_file_location(
    "_p23", _HERE / "23_global_tier_design_counts.py")
_p23 = importlib.util.module_from_spec(_spec2)
sys.modules["_p23"] = _p23
_spec2.loader.exec_module(_p23)
histify              = _p23.histify
percentile_from_hist = _p23.percentile_from_hist
CIRCUIT_LOG_RANGE    = _p23.CIRCUIT_LOG_RANGE
TOXICITY_RANGE       = _p23.TOXICITY_RANGE
N_BINS               = _p23.N_BINS
CIRCUIT_EDGES        = _p23.CIRCUIT_EDGES
TOXICITY_EDGES       = _p23.TOXICITY_EDGES
FLOOR                = _p23.FLOOR

sys.path.insert(0, str(_HERE))
from _population_filter import add_scope_args, filter_topologies, scope_summary  # noqa: E402


SIZE_CLASSES = (4, 5, 6, 7)


def _build_anchor_list() -> list[float]:
    """1003-point percentile anchor list: tail detail + 999 step-0.1."""
    anchors: list[float] = [0.01, 0.05]
    anchors += [round(i / 10.0, 1) for i in range(1, 1000)]  # 0.1 .. 99.9
    anchors += [99.95, 99.99]
    return anchors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population_manifest", type=Path, required=True)
    ap.add_argument("--qc_tiers", type=Path, required=True)
    ap.add_argument("--predictions_root", type=Path, required=True,
                    help="$DEEPCIRC_SCRATCH/topology_data/design_space_predictions"
                         " (group-agnostic pool)")
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--limit_topologies", type=int, default=None,
                    help="cap on number of topologies (debug only)")
    ap.add_argument("--include_targets", type=str, nargs="+", default=None,
                    help="Optional list of target hexes (e.g. 0x17 0x2B 0x6D) "
                         "to include. Topologies whose `source` field's first "
                         "entry is in this list are kept; all others dropped. "
                         "Default: include all targets in the manifest.")
    ap.add_argument("--output_suffix", type=str, default="",
                    help="Optional suffix appended to dense_percentile_sweep "
                         "JSON filename (e.g. '_3input'). Default: no suffix.")
    add_scope_args(ap)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    anchors_pct = _build_anchor_list()
    include_targets = None
    if args.include_targets:
        include_targets = set(t.lower() for t in args.include_targets)
    print(f"Phase 23.3 - Dense per-size percentile sweep")
    print(f"  population:      {args.population_manifest}")
    print(f"  qc_tiers:        {args.qc_tiers}")
    print(f"  predictions:     {args.predictions_root}")
    print(f"  output:          {args.output_root}")
    print(f"  size classes:    {SIZE_CLASSES}")
    print(f"  N_BINS:          {N_BINS}")
    print(f"  n_anchors:       {len(anchors_pct)}")
    print(f"  include_targets: "
          f"{sorted(include_targets) if include_targets else '(all)'}")
    print(f"  output_suffix:   '{args.output_suffix}'")
    print()

    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)
    with open(args.qc_tiers, "rb") as f:
        qc = pickle.load(f)
    qc_by_tid = {r["topology_id"]: r for r in qc["records"]}

    topologies = filter_topologies(
        pop["topologies"],
        include_baselines=args.include_baselines,
        size_classes=args.size_classes,
        label="P23.3 scope",
    )
    if args.limit_topologies is not None:
        topologies = topologies[:args.limit_topologies]

    # Aggregate histograms streamed in directly (per-size, per-task).
    size_hist_circ = {s: np.zeros(N_BINS, dtype=np.int64) for s in SIZE_CLASSES}
    size_hist_tox  = {s: np.zeros(N_BINS, dtype=np.int64) for s in SIZE_CLASSES}
    size_n_topos   = {s: 0 for s in SIZE_CLASSES}
    n_processed = n_skipped = 0
    t0 = time.perf_counter()

    for i, t in enumerate(topologies, start=1):
        tid = t["topology_id"]
        qc_rec = qc_by_tid.get(tid)
        if qc_rec is None:
            print(f"[{i}/{len(topologies)}] {tid} -- no QC record; skip")
            n_skipped += 1
            continue
        if qc_rec["tier"] in ("A3", "A0", "BROKEN"):
            print(f"[{i}/{len(topologies)}] {tid} -- SKIP (tier={qc_rec['tier']})")
            n_skipped += 1
            continue
        reg_count = int(t["regulator_count"])
        if reg_count not in SIZE_CLASSES:
            print(f"[{i}/{len(topologies)}] {tid} -- SKIP "
                  f"(regulator_count={reg_count} not in {SIZE_CLASSES})")
            n_skipped += 1
            continue
        if include_targets is not None:
            raw_source = t.get("source", "unknown")
            if isinstance(raw_source, list):
                source_targets = [str(s).lower() for s in raw_source]
            else:
                source_targets = [str(raw_source).lower()]
            # Keep if ANY source target is in the include set (multi-source
            # topologies count if they serve at least one in-scope target).
            if not any(s in include_targets for s in source_targets):
                print(f"[{i}/{len(topologies)}] {tid} -- SKIP "
                      f"(target {source_targets} not in include list)")
                n_skipped += 1
                continue

        best_perm = best_perm_for_topology(qc_rec)
        if best_perm is None:
            n_skipped += 1
            continue
        npz_path = args.predictions_root / f"{tid}_{best_perm}.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(topologies)}] {tid} -- no .npz; skip")
            n_skipped += 1
            continue

        t_start = time.perf_counter()
        try:
            d = np.load(npz_path, allow_pickle=False)
            circuit_pred  = d["circuit_pred"].astype(np.float64)
            toxicity_pred = d["toxicity_pred"].astype(np.float64)
            n_designs = circuit_pred.shape[0]

            y_circ = np.log2(np.maximum(circuit_pred, FLOOR))
            y_tox  = toxicity_pred

            size_hist_circ[reg_count] += histify(
                y_circ, CIRCUIT_EDGES, CIRCUIT_LOG_RANGE[0], CIRCUIT_LOG_RANGE[1])
            size_hist_tox[reg_count]  += histify(
                y_tox,  TOXICITY_EDGES, TOXICITY_RANGE[0], TOXICITY_RANGE[1])
            size_n_topos[reg_count]   += 1
            n_processed += 1

            elapsed = time.perf_counter() - t_start
            if i % 5 == 0 or i <= 3 or i == len(topologies):
                print(f"[{i}/{len(topologies)}] {tid} ({reg_count}-reg, "
                      f"perm={best_perm}, tier={qc_rec['tier']}) -> "
                      f"{n_designs:>10,} designs in {elapsed:.1f}s",
                      flush=True)
        except Exception as e:
            print(f"[{i}/{len(topologies)}] {tid} -- ERROR "
                  f"{type(e).__name__}: {e}")
            n_skipped += 1

    elapsed_pass1 = time.perf_counter() - t0
    print(f"\nPass 1 complete: processed={n_processed}, skipped={n_skipped}, "
          f"wall={elapsed_pass1:.1f}s")

    # -------------------------------------------------------------------
    # Dense percentile readout per (size, task)
    # -------------------------------------------------------------------
    print("\nComputing dense percentile sweep...")
    per_size_block: dict = {}
    for s in SIZE_CLASSES:
        n_designs_c = int(size_hist_circ[s].sum())
        n_designs_t = int(size_hist_tox[s].sum())
        if n_designs_c == 0 and n_designs_t == 0:
            print(f"  size {s}-reg: no designs; emitting empty block")
            per_size_block[str(s)] = {
                "n_topologies": int(size_n_topos[s]),
                "n_designs_total_circuit":  0,
                "n_designs_total_toxicity": 0,
                "circuit_log": [],
                "toxicity":    [],
            }
            continue
        circ_vec = [
            percentile_from_hist(size_hist_circ[s], CIRCUIT_EDGES, p)
            for p in anchors_pct
        ]
        tox_vec = [
            percentile_from_hist(size_hist_tox[s], TOXICITY_EDGES, p)
            for p in anchors_pct
        ]
        per_size_block[str(s)] = {
            "n_topologies": int(size_n_topos[s]),
            "n_designs_total_circuit":  n_designs_c,
            "n_designs_total_toxicity": n_designs_t,
            "circuit_log": [float(v) for v in circ_vec],
            "toxicity":    [float(v) for v in tox_vec],
        }
        # Sanity-print a few anchors
        i01 = anchors_pct.index(0.1)
        i50 = anchors_pct.index(50.0)
        i99 = anchors_pct.index(99.9)
        print(f"  size {s}-reg: n_topo={size_n_topos[s]:3d}  "
              f"n_designs_circ={n_designs_c:>12,}  "
              f"p0.1={circ_vec[i01]:+.4f}  "
              f"p50={circ_vec[i50]:+.4f}  "
              f"p99.9={circ_vec[i99]:+.4f}")

    out: dict = {
        "phase": "23_3_dense_percentile_sweep",
        "scope": scope_summary(
            include_baselines=args.include_baselines,
            size_classes=args.size_classes,
        ),
        "include_targets": (sorted(include_targets)
                            if include_targets else None),
        "n_bins": N_BINS,
        "circuit_log_range": list(CIRCUIT_LOG_RANGE),
        "toxicity_range": list(TOXICITY_RANGE),
        "elapsed_pass1_s": elapsed_pass1,
        "n_topologies_processed": n_processed,
        "n_topologies_skipped": n_skipped,
        "anchors_pct": anchors_pct,
        "per_size": per_size_block,
    }

    out_json = (args.output_root /
                f"dense_percentile_sweep{args.output_suffix}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_json}")
    sz = out_json.stat().st_size
    print(f"  file size: {sz:,} bytes ({sz / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
