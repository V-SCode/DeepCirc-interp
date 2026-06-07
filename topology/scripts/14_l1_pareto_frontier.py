"""Phase 14 — Pareto frontier (circuit margin vs growth burden) per topology.

For each trained topology in the group's substrate (A1+A2 tiers, agent-only,
4-7 reg), identify the Pareto-undominated subset of its design-space landscape
on (circuit_log, toxicity) where both axes are maximized. Pick "knee" designs
under three criteria (default A = max-perpendicular-distance-from-chord;
also emit B = max-product, C = closest-to-utopia). Aggregate to a per-topology
table and emit a cross-target buildable portfolio.

**Why this analysis**

The existing rule book emits universal rules for each axis SEPARATELY: IcaRA/I1
toxicity (one axis), first-internal-NOR circuit cluster (other axis). Biologists
need designs that win on BOTH axes simultaneously — a circuit margin worth
demonstrating without growth burden that kills the cells.

Per topology, the Pareto-undominated set is the smallest portfolio of designs
that's "as good as it gets" on the joint (circuit, toxicity) trade-off. The
knee is the design where the front "turns" most sharply — the practical pick.

**Multi-perm handling (Q3 of the spec)**

A topology has 0-6 trained input permutations (NNGGA's interference-filter
result). Each (topology, perm) is its own design space with its own MLP. We
compute per-perm Pareto fronts in memory, take the union, and recompute the
topology-level Pareto front from that union. One knee per topology.

**Inputs**

- --population_manifest  per-group population.pkl (with `group`, `topologies`)
- --predictions_root     group-agnostic per-topology pool of .npz files
                         (output_root from P7; one file per (topology, perm))
- --qc_tiers             per-group qc_tiers.pkl (used to filter to A1+A2)
- --part_response_data   $DEEPCIRC_BASE/DeepCirc_upstream/dgd/data/response_data_3_inputs_DeepCirc.json
- --output_root          per-group output (typically population/<G>/pareto/)
- --buildable_circuit_min           default 5.3; buildable circuit_score floor
                                    on the RAW (un-logged) circuit_score axis.
                                    Default = paper's weakest yellow-dot
                                    (0x6D, c=5.3); meet-or-beat anchor.
                                    Internally compared as log(buildable_circuit_min).
- --buildable_tox_min               default 0.85; buildable toxicity floor
                                    (per G2's IcaRA-toxic class convention)
- scope flags via _population_filter (agent-only, 4-7-reg defaults)

**Outputs** (per-group, under output_root)

- knee_designs.csv           one row per topology (~215 for G3)
                              cols: topology_id, source, gate_count, qc_tier,
                              n_perms_trained, n_designs_total, n_pareto_front,
                              circuit_log_max, toxicity_max,
                              knee_{A,B,C}_perm,
                              knee_{A,B,C}_circuit_log,
                              knee_{A,B,C}_toxicity,
                              knee_{A,B,C}_part_names (JSON),
                              knee_{A,B,C}_gate_assignments (JSON)
- all_topology_fronts.csv    Pareto-front rows across all topologies
                              cols: topology_id, perm_idx, circuit_log,
                              toxicity, gate_assignments (JSON),
                              part_names (JSON), source, gate_count
- cross_target_portfolio.csv buildable subset of knee_designs ranked by
                              combined score, filtered to circuit_score >=
                              5.3 (paper's weakest yellow-dot, 0x6D) AND
                              toxicity >= 0.85.
- pareto_summary.json        scope + per-NPN aggregates + buildable thresholds

**Algorithm details**

- Pareto front via skyline (sort by axis 0 desc, sweep retaining axis-1-max).
  O(N log N) per topology. Stable: ties on axis 0 are tiebroken by axis 1.
- circuit transform: m_logic = log2(max(circuit_pred, FLOOR)) for cross-figure
  log₂ consistency (matches P9 / P26 / P27 enrichment metrics; migrated from
  natural log 2026-06-01). Compresses the fat right tail of raw circuit scores
  so the Pareto front lives in a comparable [4, 13]-ish log₂ range.
- toxicity stays raw: it's already bounded in [0, 1] (simulator-faithful per
  Module A R² ≈ 0.99).
- Both axes normalized to [0, 1] before knee computation. The three knees are
  three distinct definitions of "best joint pick" — see docstrings on knee_*().

**Usage**

    python topology/scripts/14_l1_pareto_frontier.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G3/population.pkl \\
        --predictions_root    $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G3/qc_tiers.pkl \\
        --output_root         $DEEPCIRC_SCRATCH/population/G3/pareto \\
        --part_response_data  $DEEPCIRC_BASE/DeepCirc_upstream/dgd/data/response_data_3_inputs_DeepCirc.json

Compute: ~10 min wall on login CPU at G3 scale (215 topologies × avg 1.6 perms
× up-to-10M designs per perm). Memory peak ~200 MB per topology, sequential
processing. Login-node-friendly.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _population_filter import (  # noqa: E402
    add_scope_args,
    filter_topologies,
    scope_summary,
)

NUM_PARTS = 20
FLOOR = 1e-6
DEFAULT_BUILDABLE_CIRCUIT_MIN = 5.3  # paper's weakest yellow-dot (0x6D)
DEFAULT_BUILDABLE_TOX_MIN = 0.85
KEEP_TIERS = {"A1", "A2"}


# -----------------------------------------------------------------------------
# Core algorithms
# -----------------------------------------------------------------------------
def pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Return indices of Pareto-undominated rows (both axes maximized).

    Skyline: sort by axis-0 desc with axis-1 desc tiebreak, sweep retaining
    points whose axis-1 strictly exceeds the running maximum.

    O(N log N) time, O(N) space. Handles duplicates correctly: a point (a, b)
    is dominated by (a, b') iff b' > b; ties on both axes treat the
    sort-first appearance as on-front and later as dominated.
    """
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    front_mask = np.zeros(n, dtype=bool)
    max_y = -np.inf
    for idx in order:
        if points[idx, 1] > max_y:
            front_mask[idx] = True
            max_y = points[idx, 1]
    return np.where(front_mask)[0]


def normalize_to_unit(points: np.ndarray) -> np.ndarray:
    """Min-max normalize each axis to [0, 1] independently.

    Degenerate axes (constant value) collapse to 0.5 so downstream knee math
    behaves (no NaNs). Returns a new float64 array.
    """
    out = np.empty(points.shape, dtype=np.float64)
    for j in range(points.shape[1]):
        col = points[:, j].astype(np.float64)
        lo, hi = float(col.min()), float(col.max())
        rng = hi - lo
        if rng < 1e-12:
            out[:, j] = 0.5
        else:
            out[:, j] = (col - lo) / rng
    return out


def knee_perpendicular_distance(front_pts_normalized: np.ndarray) -> int:
    """Knee = max perpendicular distance from the chord connecting endpoints.

    Default Pareto-knee criterion (Option A in plan). After sorting points by
    axis-0 ascending, the chord runs from p_min_circ (high-tox / low-circ) to
    p_max_circ (high-circ / low-tox). Knee bulges most toward the (1,1) ideal
    corner. Returns the index into the unsorted input array.
    """
    n = front_pts_normalized.shape[0]
    if n == 0:
        return -1
    if n == 1:
        return 0
    order = np.argsort(front_pts_normalized[:, 0])
    sorted_pts = front_pts_normalized[order]
    p0, p1 = sorted_pts[0], sorted_pts[-1]
    line_vec = p1 - p0
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-12:
        return int(order[0])
    deltas = sorted_pts - p0
    cross = deltas[:, 0] * line_vec[1] - deltas[:, 1] * line_vec[0]
    perp = np.abs(cross) / line_len
    return int(order[int(np.argmax(perp))])


def knee_max_product(front_pts_normalized: np.ndarray) -> int:
    """Knee = max product of normalized coordinates (Option B in plan).

    Biased toward designs scoring well on BOTH axes simultaneously (toward
    the (1,1) corner). Differs from knee_perpendicular_distance for asymmetric
    Pareto fronts.
    """
    n = front_pts_normalized.shape[0]
    if n == 0:
        return -1
    return int(np.argmax(front_pts_normalized[:, 0] * front_pts_normalized[:, 1]))


def knee_closest_to_utopia(front_pts_normalized: np.ndarray) -> int:
    """Knee = minimum Euclidean distance from (1, 1) (Option C in plan).

    The "utopian point" (max circuit, max toxicity) is unattainable; the knee
    is the point on the front that fails the ideal by the least. Differs from
    knee_max_product for concave-vs-convex front shapes.
    """
    n = front_pts_normalized.shape[0]
    if n == 0:
        return -1
    diffs = front_pts_normalized - np.array([1.0, 1.0])
    dists = np.linalg.norm(diffs, axis=1)
    return int(np.argmin(dists))


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_part_names(response_data_path: Path) -> list[str]:
    """Read upstream response_data_3_inputs JSON; return 20 'Repressor/RBS'
    labels. Same convention as P9/P11/P12.
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


def list_perm_npzs(predictions_root: Path, topology_id: str) -> list[Path]:
    """Find all <tid>_<perm>.npz files in the group-agnostic pool."""
    return sorted(predictions_root.glob(f"{topology_id}_*.npz"))


def load_design_space(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load (gate_assignments, circuit_pred, toxicity_pred, meta) from an
    npz file emitted by P7 (07_score_design_space.py).

    Schema per P7 docstring:
      - gate_assignments  int8   (N, gate_count)
      - circuit_pred      float32 (N,)
      - toxicity_pred     float32 (N,)
      - meta              0d object array (dict)
    """
    d = np.load(npz_path, allow_pickle=True)
    return (
        d["gate_assignments"],
        d["circuit_pred"],
        d["toxicity_pred"],
        d["meta"].item(),
    )


# -----------------------------------------------------------------------------
# Per-topology Pareto pipeline
# -----------------------------------------------------------------------------
def per_topology_pareto(
    topology_id: str,
    perm_npzs: list[Path],
    part_names: list[str],
) -> tuple[dict, list[dict]] | tuple[None, None]:
    """For one topology, union per-perm Pareto fronts, compute topology-level
    Pareto front, pick three knees, decode part names.

    Returns (topology_summary, front_rows) where front_rows is a list of
    flat-dict rows suitable for concatenation into all_topology_fronts.csv.
    Returns (None, None) if no perms found or all empty.
    """
    if not perm_npzs:
        return None, None

    # Step 1: per-perm Pareto fronts (in memory; not emitted separately)
    aggregate_records: list[dict] = []
    n_designs_total = 0
    for npz_path in perm_npzs:
        ga, circ, tox, meta = load_design_space(npz_path)
        perm_idx = int(meta["perm_idx"])
        n_designs_total += int(ga.shape[0])

        # Log₂-transform circuit (m_logic per Module D, migrated to log₂
        # 2026-06-01 for cross-figure consistency); toxicity raw.
        circ_log = np.log2(np.maximum(circ.astype(np.float64), FLOOR))
        tox_f = tox.astype(np.float64)

        pts = np.column_stack([circ_log, tox_f])
        front_idx = pareto_front_2d(pts)

        for fi in front_idx:
            aggregate_records.append({
                "perm_idx":         perm_idx,
                "design_row_idx":   int(fi),
                "circuit_log":      float(circ_log[fi]),
                "toxicity":         float(tox_f[fi]),
                "gate_assignments": [int(x) for x in ga[fi].tolist()],
            })

    if not aggregate_records:
        return None, None

    # Step 2: topology-level Pareto front from union of per-perm fronts.
    agg_pts = np.array([[r["circuit_log"], r["toxicity"]] for r in aggregate_records])
    topo_front_idx = pareto_front_2d(agg_pts)
    topo_front_pts = agg_pts[topo_front_idx]
    topo_front_records = [aggregate_records[i] for i in topo_front_idx]

    # Step 3: three knees on topology-level front (each emitted independently)
    front_norm = normalize_to_unit(topo_front_pts)
    k_a = knee_perpendicular_distance(front_norm)
    k_b = knee_max_product(front_norm)
    k_c = knee_closest_to_utopia(front_norm)

    def _decode_knee(k_idx: int) -> dict:
        if k_idx < 0:
            return {}
        rec = topo_front_records[k_idx]
        return {
            "perm_idx":         rec["perm_idx"],
            "design_row_idx":   rec["design_row_idx"],
            "circuit_log":      rec["circuit_log"],
            "toxicity":         rec["toxicity"],
            "gate_assignments": rec["gate_assignments"],
            "part_names":       [part_names[i] for i in rec["gate_assignments"]],
        }

    summary = {
        "topology_id":              topology_id,
        "n_perms_trained":          len(perm_npzs),
        "n_designs_total":          int(n_designs_total),
        "n_perm_front_union":       len(aggregate_records),
        "n_pareto_front":           int(topo_front_idx.shape[0]),
        "circuit_log_max":          float(agg_pts[:, 0].max()),
        "toxicity_max":             float(agg_pts[:, 1].max()),
        "circuit_log_min_on_front": float(topo_front_pts[:, 0].min()),
        "toxicity_min_on_front":    float(topo_front_pts[:, 1].min()),
        "knee_A":                   _decode_knee(k_a),
        "knee_B":                   _decode_knee(k_b),
        "knee_C":                   _decode_knee(k_c),
    }

    # Flat front rows for the consolidated all_topology_fronts.csv
    front_rows = [
        {
            "topology_id":      topology_id,
            "perm_idx":         rec["perm_idx"],
            "circuit_log":      rec["circuit_log"],
            "toxicity":         rec["toxicity"],
            "gate_assignments": json.dumps(rec["gate_assignments"]),
            "part_names":       json.dumps([part_names[i] for i in rec["gate_assignments"]]),
        }
        for rec in topo_front_records
    ]

    return summary, front_rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_manifest", type=Path, required=True)
    p.add_argument("--predictions_root", type=Path, required=True,
                   help="Group-agnostic per-topology pool (post-R5 default: "
                        "$DEEPCIRC_SCRATCH/topology_data/design_space_predictions)")
    p.add_argument("--qc_tiers", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--part_response_data", type=Path, required=True,
                   help="upstream dgd/data/response_data_3_inputs_DeepCirc.json")
    p.add_argument("--buildable_circuit_min", type=float,
                   default=DEFAULT_BUILDABLE_CIRCUIT_MIN,
                   help="Buildable circuit_score floor (RAW, un-logged). "
                        "Default 5.3 = paper's weakest yellow-dot (0x6D). "
                        "Internally converted to log(circuit_min) for "
                        "comparison against knee_A_circuit_log.")
    p.add_argument("--buildable_tox_min", type=float,
                   default=DEFAULT_BUILDABLE_TOX_MIN,
                   help="Buildable toxicity floor (absolute). Default 0.85 "
                        "(per G2 IcaRA-toxic class convention).")
    add_scope_args(p)
    args = p.parse_args()

    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)
    with open(args.qc_tiers, "rb") as f:
        qc = pickle.load(f)

    group = pop.get("group", "G?")
    print(f"Phase 14 — Pareto frontier analysis for {group}")
    print(f"  population:    {args.population_manifest}")
    print(f"  predictions:   {args.predictions_root}")
    print(f"  qc_tiers:      {args.qc_tiers}")
    print(f"  output:        {args.output_root}")
    print()

    qc_by_tid = {r["topology_id"]: r for r in qc["records"]}

    pop_topos = filter_topologies(
        pop["topologies"],
        include_baselines=args.include_baselines,
        size_classes=args.size_classes,
        label="P14 scope",
    )

    part_names = load_part_names(args.part_response_data)
    print(f"[setup] loaded {len(part_names)} parts from "
          f"{args.part_response_data.name}")

    # Filter to A1+A2 trained topologies; skip A0 markers, A3 hard-to-fit.
    eligible: list[dict] = []
    skip_counts: dict[str, int] = {}
    for t in pop_topos:
        tid = t["topology_id"]
        tier = qc_by_tid.get(tid, {}).get("tier", "?")
        if tier not in KEEP_TIERS:
            skip_counts[tier] = skip_counts.get(tier, 0) + 1
            continue
        eligible.append(t)
    print(f"[setup] {len(eligible)} eligible topologies (tier in {sorted(KEEP_TIERS)}); "
          f"skipped: {skip_counts}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    all_front_rows: list[dict] = []
    t0 = time.perf_counter()
    n_eligible = len(eligible)
    for i, t in enumerate(eligible, 1):
        tid = t["topology_id"]
        perm_npzs = list_perm_npzs(args.predictions_root, tid)
        if not perm_npzs:
            print(f"[{i}/{n_eligible}] {tid} ⚠️ no .npz files in predictions_root")
            continue
        try:
            summary, front_rows = per_topology_pareto(tid, perm_npzs, part_names)
        except Exception as e:  # noqa: BLE001 (broad except: log + continue)
            print(f"[{i}/{n_eligible}] {tid} ❌ error: {type(e).__name__}: {e}")
            continue
        if summary is None:
            continue
        summary["source"]      = t.get("source")
        summary["energy"]      = t.get("energy")
        summary["gate_count"]  = t.get("regulator_count")
        summary["qc_tier"]     = qc_by_tid[tid].get("tier")
        summary["is_baseline"] = bool(t.get("is_baseline", False))
        for fr in front_rows:
            fr["source"]     = t.get("source") if not isinstance(t.get("source"), list) else json.dumps(t.get("source"))
            fr["gate_count"] = t.get("regulator_count")
        summaries.append(summary)
        all_front_rows.extend(front_rows)
        if i % 25 == 0 or i == n_eligible:
            elapsed = time.perf_counter() - t0
            print(f"[{i}/{n_eligible}] {tid} processed; running wall = "
                  f"{elapsed/60:.1f} min, fronts cum={len(all_front_rows)}")

    elapsed = time.perf_counter() - t0
    print(f"\n=== Phase 14 done in {elapsed/60:.1f} min ===")
    print(f"  topologies analyzed: {len(summaries)}")
    print(f"  Pareto-front rows (cumulative across all topologies): {len(all_front_rows)}")

    # ------------------------------------------------------------------
    # Emit knee_designs.csv (one row per topology)
    # ------------------------------------------------------------------
    knee_rows = []
    for r in summaries:
        flat = {
            "topology_id":              r["topology_id"],
            "source":                   (json.dumps(r["source"])
                                          if isinstance(r["source"], list)
                                          else r["source"]),
            "gate_count":               r["gate_count"],
            "qc_tier":                  r["qc_tier"],
            "n_perms_trained":          r["n_perms_trained"],
            "n_designs_total":          r["n_designs_total"],
            "n_perm_front_union":       r["n_perm_front_union"],
            "n_pareto_front":           r["n_pareto_front"],
            "circuit_log_max":          r["circuit_log_max"],
            "toxicity_max":             r["toxicity_max"],
            "circuit_log_min_on_front": r["circuit_log_min_on_front"],
            "toxicity_min_on_front":    r["toxicity_min_on_front"],
        }
        for tag, knee in [("A", r["knee_A"]), ("B", r["knee_B"]), ("C", r["knee_C"])]:
            flat[f"knee_{tag}_perm"]              = knee.get("perm_idx")
            flat[f"knee_{tag}_design_row_idx"]    = knee.get("design_row_idx")
            flat[f"knee_{tag}_circuit_log"]       = knee.get("circuit_log")
            flat[f"knee_{tag}_toxicity"]          = knee.get("toxicity")
            flat[f"knee_{tag}_part_names"]        = json.dumps(knee.get("part_names", []))
            flat[f"knee_{tag}_gate_assignments"]  = json.dumps(knee.get("gate_assignments", []))
        knee_rows.append(flat)
    knee_df = pd.DataFrame(knee_rows)
    knee_csv_path = args.output_root / "knee_designs.csv"
    knee_df.to_csv(knee_csv_path, index=False)
    print(f"[emit] {knee_csv_path}  ({len(knee_df)} rows)")

    # ------------------------------------------------------------------
    # Emit all_topology_fronts.csv (every topology's Pareto-front rows)
    # ------------------------------------------------------------------
    fronts_df = pd.DataFrame(all_front_rows)
    fronts_csv_path = args.output_root / "all_topology_fronts.csv"
    fronts_df.to_csv(fronts_csv_path, index=False)
    print(f"[emit] {fronts_csv_path}  ({len(fronts_df)} rows)")

    # ------------------------------------------------------------------
    # Emit cross_target_portfolio.csv (filter + rank)
    # ------------------------------------------------------------------
    circuit_log_min = None
    portfolio_size = 0
    if len(knee_df) > 0:
        circuit_log_min = float(np.log2(args.buildable_circuit_min))
        tox_min = args.buildable_tox_min
        buildable = knee_df[
            (knee_df["knee_A_circuit_log"] >= circuit_log_min)
            & (knee_df["knee_A_toxicity"] >= tox_min)
        ].copy()
        if len(buildable) > 0:
            tox_range = max(1.0 - tox_min, 1e-9)
            circ_range = max(
                float(buildable["knee_A_circuit_log"].max()) - circuit_log_min,
                1e-9,
            )
            buildable["combined_score"] = (
                (buildable["knee_A_circuit_log"] - circuit_log_min) / circ_range
                + (buildable["knee_A_toxicity"] - tox_min) / tox_range
            )
            buildable = (buildable
                         .sort_values("combined_score", ascending=False)
                         .reset_index(drop=True))
            buildable.insert(0, "rank", range(1, len(buildable) + 1))
        portfolio_path = args.output_root / "cross_target_portfolio.csv"
        buildable.to_csv(portfolio_path, index=False)
        portfolio_size = len(buildable)
        print(f"[emit] {portfolio_path}  ({portfolio_size} buildable; "
              f"circuit_score≥{args.buildable_circuit_min:.2f} "
              f"[log≥{circuit_log_min:.3f}], tox≥{tox_min:.2f})")

    # ------------------------------------------------------------------
    # Emit pareto_summary.json (scope + aggregates)
    # ------------------------------------------------------------------
    def _stats(series: pd.Series) -> dict:
        if len(series) == 0:
            return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
        return {
            "n":      int(len(series)),
            "mean":   float(series.mean()),
            "median": float(series.median()),
            "min":    float(series.min()),
            "max":    float(series.max()),
            "std":    float(series.std()),
        }

    summary_doc = {
        "scope":                 scope_summary(
            include_baselines=args.include_baselines,
            size_classes=args.size_classes,
        ),
        "group":                 group,
        "phase":                 "14_l1_pareto_frontier",
        "n_topologies_analyzed": len(summaries),
        "skipped_by_tier":       skip_counts,
        "buildable_thresholds":  {
            "circuit_score_min": float(args.buildable_circuit_min),
            "circuit_log_min":   (float(circuit_log_min) if circuit_log_min is not None else None),
            "toxicity_min":      args.buildable_tox_min,
            "anchor_rationale":  (
                "circuit_score_min = paper's weakest yellow-dot (0x6D, c=5.3); "
                "buildable = meet biological growth floor AND meet-or-beat the "
                "weakest paper-built circuit"
            ),
        },
        "portfolio_size":        portfolio_size,
        "knee_A_circuit_log":    _stats(knee_df["knee_A_circuit_log"]) if len(knee_df) else _stats(pd.Series(dtype=float)),
        "knee_A_toxicity":       _stats(knee_df["knee_A_toxicity"]) if len(knee_df) else _stats(pd.Series(dtype=float)),
        "n_pareto_front":        _stats(knee_df["n_pareto_front"]) if len(knee_df) else _stats(pd.Series(dtype=float)),
        "n_designs_total":       _stats(knee_df["n_designs_total"]) if len(knee_df) else _stats(pd.Series(dtype=float)),
        "elapsed_minutes":       elapsed / 60.0,
    }
    summary_path = args.output_root / "pareto_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_doc, f, indent=2, default=str)
    print(f"[emit] {summary_path}")


if __name__ == "__main__":
    main()
