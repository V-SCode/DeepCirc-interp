"""Phase 23.2 — Size-stratified design-tier counts (simplemotif-by-size substrate).

Variant of Phase 23 (`23_global_tier_design_counts.py`) that computes
percentile thresholds **per size class** (4-reg, 5-reg, 6-reg, 7-reg)
instead of pooling all 1.41B designs into one global histogram.

Use case: the simplemotif-by-size arc (fig29 + fig30 families) asks
"what non-linear iso-canonical motifs are most enriched in elite
designs within size class S?" — answering this requires elite tier
thresholds defined within each size class, not globally. At smaller
sizes the size-burden law means most designs would never reach the
global top-1%; per-size thresholds restore meaningful elite/failure
discrimination within each size's own design distribution.

Method mirrors P23 with one extra stratification dimension:

  1. Pass 1 (streaming): same as P23 — load each topology's
     design_space_predictions/<tid>_<perm>.npz, compute Y_circuit =
     log(circuit_pred) and Y_tox = toxicity_pred, build per-topology
     histograms over 200k bins.
  2. Aggregate per-topology histograms per SIZE CLASS → 4 size-stratified
     global histograms per task. Compute size-stratified percentile
     thresholds {p99, p95, p90, p75, p25, p10, p05, p01} from each.
  3. Per topology: count designs in each tier using the thresholds
     matching the topology's own size class.

Tier definitions per (size, task) — symmetric upper + lower tails:
  top_01_S — designs with Y >= 99th-percentile within size S
  top_05_S — designs with Y >= 95th-percentile within size S
  top_10_S — designs with Y >= 90th-percentile within size S
  top_25_S — designs with Y >= 75th-percentile within size S
  bot_25_S — designs with Y <= 25th-percentile within size S
  bot_10_S — designs with Y <= 10th-percentile within size S
  bot_05_S — designs with Y <=  5th-percentile within size S
  bot_01_S — designs with Y <=  1st-percentile within size S

Outputs (under --output_root):

  size_tier_design_counts.csv
      one row per topology with columns:
        topology_id, source, regulator_count, qc_tier, n_designs_total,
        best_perm,
        n_top_01_circuit_S, n_top_05_circuit_S, n_top_10_circuit_S,
        n_top_25_circuit_S, n_bot_25_circuit_S, n_bot_10_circuit_S,
        n_bot_05_circuit_S, n_bot_01_circuit_S,
        n_top_01_toxicity_S, n_top_05_toxicity_S, n_top_10_toxicity_S,
        n_top_25_toxicity_S, n_bot_25_toxicity_S, n_bot_10_toxicity_S,
        n_bot_05_toxicity_S, n_bot_01_toxicity_S
      (_S = size-stratified — the thresholds used match each topology's
       own size class)
  size_tier_thresholds.json
      per size class S in {4, 5, 6, 7}:
        circuit_log_p{99, 95, 90, 75, 25, 10, 05, 01}_S
        toxicity_p{99, 95, 90, 75, 25, 10, 05, 01}_S
        n_designs_total_circuit_S, n_designs_total_toxicity_S,
        n_topologies_S
      plus: scope, hist_resolution, elapsed_s, phase

Compute: ~10-15 min CPU wall total for 215 topologies (same as P23 —
single .npz pass, just one extra aggregation dimension).

Usage on cluster:

    python topology/scripts/23_2_size_tier_design_counts.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G3/population.pkl \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G3/qc_tiers.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --output_root         $DEEPCIRC_SCRATCH/population/G3/size_tier_designs

Then transfer to repo via git push from cluster:

    cp -r $DEEPCIRC_SCRATCH/population/G3/size_tier_designs \\
        $DEEPCIRC_REPO/data/topology_g3/size_tier_designs/
    cd $DEEPCIRC_REPO
    git add data/topology_g3/size_tier_designs
    git commit -m "P23.2: G3 size-stratified design-tier counts"
    git push
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
import pandas as pd

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("_p9", _HERE / "09_l2_graph_role.py")
_p9 = importlib.util.module_from_spec(_spec)
sys.modules["_p9"] = _p9
_spec.loader.exec_module(_p9)
best_perm_for_topology = _p9.best_perm_for_topology

# Re-import histogram helpers from P23 so the math is identical.
_spec2 = importlib.util.spec_from_file_location(
    "_p23", _HERE / "23_global_tier_design_counts.py")
_p23 = importlib.util.module_from_spec(_spec2)
sys.modules["_p23"] = _p23
_spec2.loader.exec_module(_p23)
histify              = _p23.histify
percentile_from_hist = _p23.percentile_from_hist
count_above          = _p23.count_above
count_below          = _p23.count_below
CIRCUIT_LOG_RANGE    = _p23.CIRCUIT_LOG_RANGE
TOXICITY_RANGE       = _p23.TOXICITY_RANGE
N_BINS               = _p23.N_BINS
CIRCUIT_EDGES        = _p23.CIRCUIT_EDGES
TOXICITY_EDGES       = _p23.TOXICITY_EDGES
FLOOR                = _p23.FLOOR

sys.path.insert(0, str(_HERE))
from _population_filter import add_scope_args, filter_topologies, scope_summary  # noqa: E402


SIZE_CLASSES = (4, 5, 6, 7)


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
    add_scope_args(ap)
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Phase 23.2 - Size-stratified design-tier counts")
    print(f"  population:    {args.population_manifest}")
    print(f"  qc_tiers:      {args.qc_tiers}")
    print(f"  predictions:   {args.predictions_root}")
    print(f"  output:        {args.output_root}")
    print(f"  size classes:  {SIZE_CLASSES}")
    print(f"  N_BINS:        {N_BINS}")
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
        label="P23.2 scope",
    )
    if args.limit_topologies is not None:
        topologies = topologies[:args.limit_topologies]

    # Per-topology histograms saved for the recounting step
    per_topo_hists: dict[str, dict[str, np.ndarray]] = {}
    per_topo_log: list[dict] = []
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

            hist_circ = histify(y_circ, CIRCUIT_EDGES,
                                 CIRCUIT_LOG_RANGE[0], CIRCUIT_LOG_RANGE[1])
            hist_tox  = histify(y_tox,  TOXICITY_EDGES,
                                 TOXICITY_RANGE[0],    TOXICITY_RANGE[1])

            per_topo_hists[tid] = {
                "circuit_log":     hist_circ,
                "toxicity":        hist_tox,
                "n_designs":       n_designs,
                "regulator_count": reg_count,
            }

            raw_source = t.get("source", "unknown")
            if not isinstance(raw_source, list):
                raw_source = [raw_source] if raw_source else ["unknown"]
            per_topo_log.append({
                "topology_id":     tid,
                "source":          json.dumps(raw_source),
                "regulator_count": reg_count,
                "qc_tier":         qc_rec["tier"],
                "n_designs_total": int(n_designs),
                "best_perm":       int(best_perm),
            })
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
    # Size-stratified histogram aggregation + percentile thresholds
    # -------------------------------------------------------------------
    print("\nAggregating size-stratified histograms...")
    size_hist_circ = {s: np.zeros(N_BINS, dtype=np.int64) for s in SIZE_CLASSES}
    size_hist_tox  = {s: np.zeros(N_BINS, dtype=np.int64) for s in SIZE_CLASSES}
    size_n_topos   = {s: 0 for s in SIZE_CLASSES}
    for tid, h in per_topo_hists.items():
        s = h["regulator_count"]
        if s not in SIZE_CLASSES:
            continue
        size_hist_circ[s] += h["circuit_log"]
        size_hist_tox[s]  += h["toxicity"]
        size_n_topos[s]   += 1

    PCTS = [99.0, 95.0, 90.0, 75.0, 25.0, 10.0, 5.0, 1.0]
    PCT_LABELS = ["p99", "p95", "p90", "p75", "p25", "p10", "p05", "p01"]
    thresholds_circ: dict[int, dict[str, float]] = {}
    thresholds_tox:  dict[int, dict[str, float]] = {}
    total_circ_by_size: dict[int, int] = {}
    total_tox_by_size:  dict[int, int] = {}
    for s in SIZE_CLASSES:
        thresholds_circ[s] = {}
        thresholds_tox[s]  = {}
        for pct, lbl in zip(PCTS, PCT_LABELS):
            thresholds_circ[s][lbl] = percentile_from_hist(
                size_hist_circ[s], CIRCUIT_EDGES, pct)
            thresholds_tox[s][lbl]  = percentile_from_hist(
                size_hist_tox[s],  TOXICITY_EDGES, pct)
        total_circ_by_size[s] = int(size_hist_circ[s].sum())
        total_tox_by_size[s]  = int(size_hist_tox[s].sum())
        print(f"  size {s}-reg: n_topo={size_n_topos[s]:3d}  "
              f"n_designs_circ={total_circ_by_size[s]:>12,}  "
              f"p01={thresholds_circ[s]['p01']:+.4f}  "
              f"p05={thresholds_circ[s]['p05']:+.4f}  "
              f"p25={thresholds_circ[s]['p25']:+.4f}  "
              f"p75={thresholds_circ[s]['p75']:+.4f}  "
              f"p95={thresholds_circ[s]['p95']:+.4f}  "
              f"p99={thresholds_circ[s]['p99']:+.4f}")
        print(f"             tox: "
              f"p01={thresholds_tox[s]['p01']:+.4f}  "
              f"p05={thresholds_tox[s]['p05']:+.4f}  "
              f"p25={thresholds_tox[s]['p25']:+.4f}  "
              f"p75={thresholds_tox[s]['p75']:+.4f}  "
              f"p95={thresholds_tox[s]['p95']:+.4f}  "
              f"p99={thresholds_tox[s]['p99']:+.4f}")

    # -------------------------------------------------------------------
    # Per-topology tier counts using thresholds matching its own size
    # -------------------------------------------------------------------
    print("\nCounting per-topology tier membership (size-stratified)...")
    rows = list(per_topo_log)
    by_tid = {r["topology_id"]: r for r in rows}
    for tid, h in per_topo_hists.items():
        r = by_tid[tid]
        s = h["regulator_count"]
        if s not in SIZE_CLASSES:
            continue
        tc = thresholds_circ[s]
        tt = thresholds_tox[s]
        r["n_top_01_circuit_S"] = count_above(h["circuit_log"], CIRCUIT_EDGES, tc["p99"])
        r["n_top_05_circuit_S"] = count_above(h["circuit_log"], CIRCUIT_EDGES, tc["p95"])
        r["n_top_10_circuit_S"] = count_above(h["circuit_log"], CIRCUIT_EDGES, tc["p90"])
        r["n_top_25_circuit_S"] = count_above(h["circuit_log"], CIRCUIT_EDGES, tc["p75"])
        r["n_bot_25_circuit_S"] = count_below(h["circuit_log"], CIRCUIT_EDGES, tc["p25"])
        r["n_bot_10_circuit_S"] = count_below(h["circuit_log"], CIRCUIT_EDGES, tc["p10"])
        r["n_bot_05_circuit_S"] = count_below(h["circuit_log"], CIRCUIT_EDGES, tc["p05"])
        r["n_bot_01_circuit_S"] = count_below(h["circuit_log"], CIRCUIT_EDGES, tc["p01"])
        r["n_top_01_toxicity_S"] = count_above(h["toxicity"], TOXICITY_EDGES, tt["p99"])
        r["n_top_05_toxicity_S"] = count_above(h["toxicity"], TOXICITY_EDGES, tt["p95"])
        r["n_top_10_toxicity_S"] = count_above(h["toxicity"], TOXICITY_EDGES, tt["p90"])
        r["n_top_25_toxicity_S"] = count_above(h["toxicity"], TOXICITY_EDGES, tt["p75"])
        r["n_bot_25_toxicity_S"] = count_below(h["toxicity"], TOXICITY_EDGES, tt["p25"])
        r["n_bot_10_toxicity_S"] = count_below(h["toxicity"], TOXICITY_EDGES, tt["p10"])
        r["n_bot_05_toxicity_S"] = count_below(h["toxicity"], TOXICITY_EDGES, tt["p05"])
        r["n_bot_01_toxicity_S"] = count_below(h["toxicity"], TOXICITY_EDGES, tt["p01"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["regulator_count", "topology_id"]).reset_index(drop=True)

    out_csv = args.output_root / "size_tier_design_counts.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows)")

    thresholds_json: dict = {
        "phase": "23_2_size_tier_design_counts",
        "scope": scope_summary(
            include_baselines=args.include_baselines,
            size_classes=args.size_classes,
        ),
        "n_bins": N_BINS,
        "circuit_log_range": list(CIRCUIT_LOG_RANGE),
        "toxicity_range": list(TOXICITY_RANGE),
        "elapsed_pass1_s": elapsed_pass1,
        "n_topologies_processed": n_processed,
        "n_topologies_skipped": n_skipped,
        "per_size": {},
    }
    for s in SIZE_CLASSES:
        per_size_block: dict = {
            "n_topologies": int(size_n_topos[s]),
            "n_designs_total_circuit": int(total_circ_by_size[s]),
            "n_designs_total_toxicity": int(total_tox_by_size[s]),
        }
        for lbl in PCT_LABELS:
            per_size_block[f"circuit_log_{lbl}"] = float(thresholds_circ[s][lbl])
            per_size_block[f"toxicity_{lbl}"]    = float(thresholds_tox[s][lbl])
        thresholds_json["per_size"][str(s)] = per_size_block

    out_json = args.output_root / "size_tier_thresholds.json"
    with open(out_json, "w") as f:
        json.dump(thresholds_json, f, indent=2)
    print(f"wrote {out_json}")

    # Sanity: each tier's share within its size class should match nominal %
    print(f"\nSanity (per-size, per-tier share / size's own total):")
    for s in SIZE_CLASSES:
        df_s = df[df["regulator_count"] == s]
        tot_c = float(total_circ_by_size[s])
        tot_t = float(total_tox_by_size[s])
        if tot_c <= 0:
            print(f"  size {s}-reg: no designs; skip")
            continue
        for tier_pct, col_circ, col_tox in [
            (0.01, "n_top_01_circuit_S", "n_top_01_toxicity_S"),
            (0.05, "n_top_05_circuit_S", "n_top_05_toxicity_S"),
            (0.25, "n_top_25_circuit_S", "n_top_25_toxicity_S"),
            (0.25, "n_bot_25_circuit_S", "n_bot_25_toxicity_S"),
            (0.05, "n_bot_05_circuit_S", "n_bot_05_toxicity_S"),
            (0.01, "n_bot_01_circuit_S", "n_bot_01_toxicity_S"),
        ]:
            share_c = df_s[col_circ].sum() / tot_c
            share_t = df_s[col_tox].sum()  / tot_t
            print(f"  size {s}-reg  expected ~{tier_pct:.2%}   "
                  f"{col_circ}={share_c:.2%}   {col_tox}={share_t:.2%}")


if __name__ == "__main__":
    main()
