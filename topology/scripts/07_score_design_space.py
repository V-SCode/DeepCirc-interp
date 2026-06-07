"""Phase 7 — MLP-labeled design space.

For each (topology, perm) trained MLP pair (excluding A0 markers), forward-pass
both `circuit_score_model.pt` and `toxicity_score_model.pt` over the *full*
valid-K part-assignment space and save predicted (circuit, growth) per design.

Why this is needed even though NNGGA already saved
`gate_assignments_predicted_to_have_high_scores_*.npy`: that .npy holds only
the MLP-top-K survivors after the interference filter (~11% of valid-K for
4-reg, smaller fractions at higher gate_count). L1+ analyses
(Shapley over slots, ANOVA decomposition, top-vs-bottom enrichment) need
predictions over the *full* design landscape.

Inputs (paths shown post-R5; pre-R5 paths in Usage block at bottom):
  - <run_root>/<topology_id>/<topology_id>_design_0_permutation_<j>/{circuit,toxicity}_score_model.pt
  - For gate_count <= 6:
      <run_root>/<topology_id>/gate_permutations/valid_permutations_<gate_count>_gates_<...>.h5
    (NNGGA-cached, contains *all* valid permutations as int rows of length gate_count)
  - For gate_count >= 7:
      <perm_dir>/sampled_valid_permutations_<gate_count>_gates_10000000_samples.h5
    (NNGGA-cached, ~10M sampled valid permutations — typically >90% of valid-K)
  - Optional QC tier file (qc_tiers.pkl from P5) — used to skip A3 + A0 +
    BROKEN topologies if --qc_tiers is passed. (Per-group input.)

Output: <output_root>/<topology_id>_<perm_idx>.npz
with keys:
  - gate_assignments  int8   shape (N, gate_count)
  - circuit_pred      float32 shape (N,)
  - toxicity_pred     float32 shape (N,)
  - meta              0d object array  (gate_count, source, energy, tier)

Path convention (Option C — group-agnostic per-topology pool; canonical
layout in scripts/_paths.py):

    `--run_root` and `--output_root` are GROUP-AGNOSTIC. Both should
    point at the shared per-topology pool when the post-R5 migration
    has run. Then G2's P7 (cumulative 10-target substrate) reads G1's
    existing trained MLPs from the shared run_root and writes/reuses
    .npz files in the shared output_root. Idempotency works file-level
    (out_path.exists() check below) so existing .npz files from earlier
    groups are reused without recomputation. ~50 G1-overlapping
    topologies are skipped under G2; only G2-new topologies' design
    spaces get scored fresh.

Idempotent: skips per-(topology, perm) if the .npz already exists. The
file-level skip semantics make this safe even when output_root points
at a shared cross-group pool.

Compute: GPU-friendly (~seconds for 4-6-reg, minutes for 7+-reg with 10M
samples). On CPU it's ~10× slower but still tractable.

Usage (post-R5 group-agnostic — recommended for G2/G3 fires):
    python 07_score_design_space.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G2/population.pkl \\
        --run_root            $DEEPCIRC_SCRATCH/runs/stage2 \\
        --output_root         $DEEPCIRC_SCRATCH/topology_data/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G2/qc_tiers.pkl

Usage (legacy per-group — pre-R5 G1 layout):
    python 07_score_design_space.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G1/population.pkl \\
        --run_root            $DEEPCIRC_SCRATCH/runs/stage2/G1 \\
        --output_root         $DEEPCIRC_SCRATCH/population/G1/design_space_predictions \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G1/qc_tiers.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Set this BEFORE importing any DGD modules; matches the wrapper / NNGGA pattern
# (PyTorch 2.0+ default validate_args=True chokes on sb3-contrib's masked-categorical
# distributions even though we don't use them here).
torch.distributions.Distribution.set_default_validate_args(False)

from dgd.utils.nnassignments import RegressionNN  # noqa: E402

from _population_filter import add_scope_args, filter_topologies  # noqa: E402

NUM_PARTS = 20  # part library size (constant across the project)
ZERO_PERMS_MARKER = "_zero_perms_filtered.json"


def load_mlp(state_dict_path: Path, gate_count: int) -> torch.nn.Module:
    """Instantiate RegressionNN with NNGGA's training hyperparams + load weights.

    Architecture must match NNGGA's per `dgd.utils.nnassignments.RegressionNN`:
      input_size = gate_count * 20  (one-hot of part-assignment)
      hidden_size = 100
      output_size = 1
      num_layers = 9
      dropout = 0.0
      activation = nn.ReLU
    """
    input_size = gate_count * NUM_PARTS
    model = RegressionNN(
        input_size=input_size,
        hidden_size=100,
        output_size=1,
        num_layers=9,
        dropout=0.0,
        activation=torch.nn.ReLU,
    )
    sd = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


def load_valid_perms(topology_dir: Path, perm_dir: Path,
                     gate_count: int) -> np.ndarray:
    """Locate + load NNGGA's cached valid_permutations h5.

    For gate_count <= 6 → topology_dir/gate_permutations/valid_permutations_<...>.h5
    (full enumeration; same file shared across all perm dirs of this topology).

    For gate_count >= 7 → perm_dir/sampled_valid_permutations_<...>.h5
    (~10M samples drawn for that perm dir specifically; NNGGA samples per
    perm with a fresh seed, so different perm dirs hold *different* samples
    of the same valid-K distribution. Statistically equivalent for our
    aggregate analyses).

    Returns int64 array of shape (N_designs, gate_count).
    """
    if gate_count <= 6:
        candidates = list((topology_dir / "gate_permutations").glob(
            f"valid_permutations_{gate_count}_gates_*.h5"))
    else:
        candidates = list(perm_dir.glob(
            f"sampled_valid_permutations_{gate_count}_gates_*.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"no valid_permutations h5 found "
            f"(gate_count={gate_count}, topology_dir={topology_dir}, "
            f"perm_dir={perm_dir})"
        )
    if len(candidates) > 1:
        # If multiple files exist (shouldn't, but defensive), prefer largest
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    df = pd.read_hdf(candidates[0], key="df")
    return df.to_numpy(dtype=np.int64)


def score_designs(perms: np.ndarray, circuit_mlp: torch.nn.Module,
                  toxicity_mlp: torch.nn.Module, device: torch.device,
                  batch_size: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Forward-pass both MLPs over all designs in one-hot batches.

    Returns (circuit_pred float32 [N], toxicity_pred float32 [N]).
    """
    circuit_mlp = circuit_mlp.to(device)
    toxicity_mlp = toxicity_mlp.to(device)
    n, gate_count = perms.shape
    circuit_out = np.empty(n, dtype=np.float32)
    toxicity_out = np.empty(n, dtype=np.float32)
    perms_t = torch.from_numpy(perms.astype(np.int64))
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = perms_t[i:i + batch_size]
            X = F.one_hot(batch, num_classes=NUM_PARTS).float().flatten(1).to(device)
            circuit_out[i:i + batch_size] = circuit_mlp(X).squeeze(-1).cpu().numpy()
            toxicity_out[i:i + batch_size] = toxicity_mlp(X).squeeze(-1).cpu().numpy()
    return circuit_out, toxicity_out


def process_topology(topology: dict, run_root: Path, output_root: Path,
                     device: torch.device, qc_tier: str | None) -> dict:
    """Score every trained perm dir for one topology; save .npz per perm.

    Returns a result record summarizing what was done / skipped / errored.
    """
    tid = topology["topology_id"]
    cn = topology.get("nngga_circuit_name", tid)
    gate_count = topology["regulator_count"]
    final_dir = run_root / tid

    if not final_dir.is_dir():
        return {"topology_id": tid, "status": "no_final_dir"}
    if (final_dir / ZERO_PERMS_MARKER).exists():
        return {"topology_id": tid, "status": "marker_skip"}

    perm_dirs = sorted(final_dir.glob(f"{cn}_design_0_permutation_*"))
    perm_results: list[dict] = []

    for pd_ in perm_dirs:
        perm_idx = int(pd_.name.split("_")[-1])
        circuit_pt = pd_ / "circuit_score_model.pt"
        tox_pt = pd_ / "toxicity_score_model.pt"
        if not (circuit_pt.exists() and tox_pt.exists()):
            perm_results.append({"perm_idx": perm_idx, "status": "untrained_skip"})
            continue

        out_path = output_root / f"{tid}_{perm_idx}.npz"
        if out_path.exists():
            perm_results.append({"perm_idx": perm_idx, "status": "already_done",
                                 "out_path": str(out_path)})
            continue

        t0 = time.perf_counter()
        try:
            perms = load_valid_perms(final_dir, pd_, gate_count)
        except FileNotFoundError as e:
            perm_results.append({"perm_idx": perm_idx, "status": f"error: {e}"})
            continue

        circuit_mlp = load_mlp(circuit_pt, gate_count)
        tox_mlp = load_mlp(tox_pt, gate_count)
        circuit_pred, tox_pred = score_designs(perms, circuit_mlp, tox_mlp, device)
        elapsed = time.perf_counter() - t0

        np.savez_compressed(
            out_path,
            gate_assignments=perms.astype(np.int8),
            circuit_pred=circuit_pred,
            toxicity_pred=tox_pred,
            meta=np.array({
                "topology_id":     tid,
                "nngga_circuit_name": cn,
                "gate_count":      gate_count,
                "perm_idx":        perm_idx,
                "source":          topology.get("source"),
                "energy":          topology.get("energy"),
                "is_baseline":     topology.get("is_baseline"),
                "is_parser_pick":  topology.get("is_parser_pick"),
                "qc_tier":         qc_tier,
                "n_designs":       int(perms.shape[0]),
                "circuit_pred_min": float(circuit_pred.min()),
                "circuit_pred_max": float(circuit_pred.max()),
                "toxicity_pred_min": float(tox_pred.min()),
                "toxicity_pred_max": float(tox_pred.max()),
            }, dtype=object),
        )
        perm_results.append({
            "perm_idx":  perm_idx,
            "status":    "ok",
            "n_designs": int(perms.shape[0]),
            "elapsed_s": elapsed,
            "out_path":  str(out_path),
        })

    return {"topology_id": tid, "status": "ok", "perm_results": perm_results}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_manifest", type=Path, required=True)
    p.add_argument("--run_root", type=Path, required=True,
                   help="Path to runs/stage2/G1 (or G2/G3 in future groups)")
    p.add_argument("--output_root", type=Path, required=True,
                   help="Per-(topology, perm) .npz lands here")
    p.add_argument("--qc_tiers", type=Path, default=None,
                   help="Optional path to qc_tiers.pkl. If passed, A3/A0/BROKEN "
                        "topologies are skipped (only A1/A2 get design-space "
                        "predictions, matching the §3 plan).")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' / 'cuda' / 'cpu'")
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--limit_topologies", type=int, default=None,
                   help="If set, only process the first N topologies (for testing)")
    add_scope_args(p)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}  (cuda available: {torch.cuda.is_available()})")

    args.output_root.mkdir(parents=True, exist_ok=True)

    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)

    # Map topology_id -> tier from qc_tiers.pkl, optionally
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
        label="P7 scope",
    )
    if args.limit_topologies is not None:
        topologies = topologies[:args.limit_topologies]
        print(f"[limit] processing only first {len(topologies)} topologies")

    n_done = n_skipped = n_already = n_error = n_perms_done = 0
    total_elapsed = 0.0
    t0 = time.perf_counter()

    for i, t in enumerate(topologies, start=1):
        tid = t["topology_id"]
        if tid in skip_ids:
            print(f"[{i}/{len(topologies)}] {tid} ({t['regulator_count']}-reg) "
                  f"-- SKIP (tier={tier_for_topology.get(tid, '?')})")
            n_skipped += 1
            continue

        print(f"[{i}/{len(topologies)}] {tid} ({t['regulator_count']}-reg, "
              f"src={t['source']}, tier={tier_for_topology.get(tid, '?')})")
        result = process_topology(t, args.run_root, args.output_root, device,
                                   tier_for_topology.get(tid))
        if result["status"] == "marker_skip":
            print(f"  marker -> skipping")
            n_skipped += 1
            continue
        if result["status"] == "no_final_dir":
            print(f"  WARN: no final dir")
            n_error += 1
            continue
        for pr in result["perm_results"]:
            if pr["status"] == "ok":
                print(f"  perm {pr['perm_idx']:>2}: {pr['n_designs']:>10,} designs "
                      f"in {pr['elapsed_s']:.1f}s")
                n_perms_done += 1
                total_elapsed += pr["elapsed_s"]
            elif pr["status"] == "already_done":
                print(f"  perm {pr['perm_idx']:>2}: already_done")
                n_already += 1
            else:
                print(f"  perm {pr['perm_idx']:>2}: {pr['status']}")
                if pr["status"].startswith("error"):
                    n_error += 1
        n_done += 1

    print()
    print(f"=== Phase 7 done in {(time.perf_counter() - t0)/60:.1f} min ===")
    print(f"  topologies processed: {n_done}")
    print(f"  skipped (A0/A3/marker): {n_skipped}")
    print(f"  perm-MLPs scored: {n_perms_done}")
    print(f"  perm-MLPs already done: {n_already}")
    print(f"  errors: {n_error}")
    print(f"  total scoring time (sum across perms): {total_elapsed/60:.1f} min")
    print(f"  output dir: {args.output_root}")


if __name__ == "__main__":
    main()
