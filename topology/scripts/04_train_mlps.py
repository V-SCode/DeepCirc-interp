"""Phase 4 — per-topology Stage-2 MLP training (idempotent + atomic).

For one population entry (selected by index from population.pkl), invoke
upstream's `NNGGA designs parallel GPUs.py` to:

  1. Generate simulator labels for ~10% of the topology's valid permutations
  2. Train circuit_score MLP (25 epochs, hidden 100 × 9 layers, ReLU, MSE, Adam)
  3. Train growth/toxicity_score MLP (same hyperparameters)
  4. Save per-topology checkpoints + metrics

All hyperparameters match upstream defaults per PROJECT_STATE_TOPOLOGY.md §2.1
("match upstream DeepCirc defaults exactly").

Path convention (R1 onwards — Option C group-agnostic per-topology pool):

    Pass `--output_root $DEEPCIRC_SCRATCH/runs/stage2`

Per-topology MLPs land at `<output_root>/<topology_id>/`. The pool is GROUP-
AGNOSTIC: a topology's MLPs are computed once and reused across any group
whose substrate contains it (G1 / G2 / G3). Idempotency check skips
already-trained topologies, so re-running P4 under G2 is a no-op for the
~50 topologies that overlap with G1's substrate; only NEW topologies in
G2's cumulative substrate get trained.

Pre-R5 (legacy v1.x / v2.0 G1) path: `--output_root .../runs/stage2/G1`
gave per-group dirs at `runs/stage2/G1/<topology_id>/`. The R5 migration
script moves these to the group-agnostic root. After migration, the new
default convention applies; pre-migration G1 data still works if the user
points --output_root at the legacy per-group dir.

Idempotency + atomicity for SLURM `--requeue` safety:
  - Before running NNGGA, we check whether the per-topology output dir already
    contains both circuit_score_model.pt and toxicity_score_model.pt with
    non-empty content. If yes → skip and exit 0.
  - NNGGA writes to a `.tmp` work dir; on success we rename to the final dir.
    If preempted mid-run, the `.tmp` dir is incomplete; the next array task
    won't see the final dir and will re-run.

**Input permutation handling:** Per the upstream paper's pipeline, NNGGA iterates
all 6 input-mapping permutations (Ara/aTc/IPTG → A/B/C) by default and trains
one (circuit_MLP, growth_MLP) pair per permutation. The published yellow-dots
for 0x2B/0x17/0x6D are the best design across all 6 perms. To match this, the
default below omits `--permutation_indices`, giving 6 MLP pairs per topology.

For experimentation/debug runs where you want only the canonical perm
(Ara→A, aTc→B, IPTG→C), pass `--permutation_indices 0` explicitly. This
reduces compute 6× but deviates from upstream paper methodology.

Usage (post-R5 group-agnostic — recommended):
    python 04_train_mlps.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G2/population.pkl \\
        --topology_index $SLURM_ARRAY_TASK_ID \\
        --output_root $DEEPCIRC_SCRATCH/runs/stage2

Usage (legacy per-group — for pre-R5 G1 data only):
    python 04_train_mlps.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G1/population.pkl \\
        --topology_index $SLURM_ARRAY_TASK_ID \\
        --output_root $DEEPCIRC_SCRATCH/runs/stage2/G1
"""
from __future__ import annotations

import torch
torch.distributions.Distribution.set_default_validate_args(False)

import argparse
import os
import pickle
import runpy
import shutil
import sys
import time
from pathlib import Path

DEEPCIRC_BASE = Path(os.environ.get("DEEPCIRC_BASE", "~/deepcirc")).expanduser()
UPSTREAM = DEEPCIRC_BASE / "DeepCirc_upstream"
NNGGA_SCRIPT = UPSTREAM / "scripts" / "NNGGA designs parallel GPUs.py"
DATA_DIR = UPSTREAM / "dgd" / "data"


ZERO_PERMS_MARKER = "_zero_perms_filtered.json"


def _is_trained_perm_dir(d: Path) -> bool:
    """True iff perm_dir has both circuit + growth MLP checkpoints non-empty.

    Distinguishes "trained" perm_dirs (NNGGA's per-perm training succeeded)
    from "interference-rejected" perm_dirs (NNGGA's >=7-reg branch creates
    the perm_dir + writes the sample h5 *before* the interference check at
    NNGGA L414, then `continue`s without training when interference trips —
    leaving a perm_dir with only the h5, no .pt files). The latter are an
    expected side effect of NNGGA's eager dir creation, not a failure.
    """
    circuit_pt = d / "circuit_score_model.pt"
    tox_pt = d / "toxicity_score_model.pt"
    return (circuit_pt.exists() and circuit_pt.stat().st_size > 0
            and tox_pt.exists() and tox_pt.stat().st_size > 0)


def already_trained(output_dir: Path, circuit_name: str) -> bool:
    """True iff output_dir holds a complete NNGGA run.

    A "complete" run is either:
      (a) ≥1 `<circuit_name>_design_0_permutation_*` subdir contains both
          `circuit_score_model.pt` and `toxicity_score_model.pt` non-empty
          (sibling perm_dirs without .pt files are interference-rejected,
          which is fine — see `_is_trained_perm_dir` docstring); OR
      (b) the dir contains a `_zero_perms_filtered.json` marker — meaning
          NNGGA returned cleanly with 0 input perms surviving its filters.

    Why the variable perm count: NNGGA filters the 6 input-permutations of
    (Ara, aTc, IPTG) → (A, B, C) in two stages:

      1. Truth-table preservation — kept iff `calculate_truth_table_v2` of
         the input-permuted graph equals that of the original (NNGGA L162-170).
      2. Interference filter — skipped iff `is_interference(...)` flags the
         topology under that input mapping (NNGGA L414-416).

    Trained-perm count after both filters ranges 0..6. We only require
    >=1 trained perm_dir to consider the topology done; the marker handles
    the all-filtered edge case.

    Preemption safety: sound because runpy.run_path() returning cleanly
    means NNGGA finished its perm loop; if preempted, runpy raises and the
    wrapper never reaches the post-run check, leaving tmp_dir to be wiped
    on retry.
    """
    if not output_dir.exists():
        return False
    if (output_dir / ZERO_PERMS_MARKER).exists():
        return True
    perm_dirs = sorted(output_dir.glob(f"{circuit_name}_design_0_permutation_*"))
    return any(_is_trained_perm_dir(d) for d in perm_dirs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_manifest", type=Path, required=True,
                   help="Path to population.pkl produced by Phase 3")
    p.add_argument("--topology_index", type=int, required=True,
                   help="Which entry in population.topologies to train")
    p.add_argument("--output_root", type=Path, required=True,
                   help="Root output dir; per-topology subdirs are created within")

    # Hyperparameters — defaults match upstream per PROJECT_STATE_TOPOLOGY.md §2.1
    p.add_argument("--percentage", type=float, default=0.10)
    p.add_argument("--num_epochs_circuit_score_model", type=int, default=25)
    p.add_argument("--num_epochs_toxicity_score_model", type=int, default=25)
    p.add_argument("--inputs_with_interference", nargs="+", default=["0", "2"])
    p.add_argument("--regulators_with_interference", nargs="+",
                   default=["PhlF", "SrpR", "BM3R1", "QacR"])
    p.add_argument("--interference_evaluation_excluded_node_types", nargs="+",
                   default=["input"])
    p.add_argument("--permutation_indices", nargs="+", default=None,
                   help="Which input permutations to train MLPs for. Default None: "
                        "match upstream paper behavior — iterate all 6 input mappings, "
                        "yielding 6 MLP pairs per topology. Pass specific indices "
                        "(e.g. '--permutation_indices 0') to restrict.")
    p.add_argument("--no_plotting", action="store_true",
                   help="Skip NNGGA's plotting step (faster)")
    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Load population manifest, pick the target entry
    # -------------------------------------------------------------------------
    if not args.population_manifest.exists():
        raise SystemExit(f"population manifest not found: {args.population_manifest}")
    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)

    n_topologies = len(pop["topologies"])
    if not (0 <= args.topology_index < n_topologies):
        raise SystemExit(f"topology_index {args.topology_index} out of range [0, {n_topologies})")
    entry = pop["topologies"][args.topology_index]

    topology_id = entry["topology_id"]
    circuit_name = entry["nngga_circuit_name"]
    nngga_pkl = Path(entry["nngga_pkl"])
    energy = entry["energy"]
    is_baseline = entry["is_baseline"]
    source = entry["source"]

    # 3-input data files, matching upstream demo
    input_data = DATA_DIR / "input_data_3_inputs_DeepCirc.json"
    response_data = DATA_DIR / "response_data_3_inputs_DeepCirc.json"
    growth_data = DATA_DIR / "growth_data_3_inputs_DeepCirc.json"
    constraints = DATA_DIR / "part_max_incoming_signals_3_inputs.json"

    for f in (NNGGA_SCRIPT, nngga_pkl, input_data, response_data, growth_data, constraints):
        if not Path(f).exists():
            raise SystemExit(f"missing required file: {f}")

    final_dir = args.output_root / topology_id
    tmp_dir = args.output_root / f"{topology_id}.tmp"

    print("=" * 64)
    print(f"Phase 4 — MLP training for topology {topology_id} "
          f"(group={pop['group']}, idx={args.topology_index} of {n_topologies})")
    print("=" * 64)
    print(f"  source          : {source}")
    print(f"  is_baseline     : {is_baseline}")
    print(f"  energy          : {energy}")
    print(f"  num_nodes       : {entry['num_nodes']}")
    print(f"  num_edges       : {entry['num_edges']}")
    print(f"  depth           : {entry['depth']}")
    print(f"  nngga_pkl       : {nngga_pkl}")
    print(f"  output (final)  : {final_dir}")
    print(f"  output (tmp)    : {tmp_dir}")

    # -------------------------------------------------------------------------
    # Idempotency: skip if final_dir is already a complete NNGGA run
    # (NNGGA's surviving perm count varies per topology — see already_trained
    # docstring for the truth-table + interference filters that drive this.)
    # -------------------------------------------------------------------------
    if already_trained(final_dir, circuit_name):
        n_perms = len(sorted(final_dir.glob(f"{circuit_name}_design_0_permutation_*")))
        print()
        print(f"[idempotency] {topology_id}: complete with {n_perms} perm dir(s) in {final_dir}")
        print(f"[idempotency] skipping (delete the dir to force re-run)")
        return

    # -------------------------------------------------------------------------
    # Atomic write: stage outputs in tmp_dir, rename on success
    # -------------------------------------------------------------------------
    recovered_from_tmp = False
    if tmp_dir.exists():
        if already_trained(tmp_dir, circuit_name):
            # Prior run completed NNGGA but failed the (now-fixed) wrapper
            # post-check. The trained MLPs in tmp_dir are valid; recover them
            # without re-running NNGGA.
            n_t = sum(1 for d in tmp_dir.glob(f"{circuit_name}_design_0_permutation_*")
                      if _is_trained_perm_dir(d))
            print(f"[recover] tmp_dir already has {n_t} trained perm dir(s); "
                  f"promoting without re-running NNGGA")
            recovered_from_tmp = True
        else:
            print(f"[atomicity] tmp_dir exists from a previous failed/preempted run — "
                  f"removing and starting fresh")
            shutil.rmtree(tmp_dir)
    if not recovered_from_tmp:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Run NNGGA (skip if recovering valid MLPs from a prior tmp_dir).
    # -------------------------------------------------------------------------
    if recovered_from_tmp:
        elapsed = 0.0
        print(f"[recover] skipping NNGGA invocation; using {tmp_dir} as-is")
    else:
        nngga_argv = [
            "NNGGA designs parallel GPUs",
            "--circuit_file_path",                            str(nngga_pkl),
            "--circuit_name",                                 circuit_name,
            "--output_dir",                                   str(tmp_dir),
            "--input_data_path",                              str(input_data),
            "--response_data_file_path",                      str(response_data),
            "--growth_data_file_path",                        str(growth_data),
            "--gate_max_incoming_signals_path",               str(constraints),
            "--inputs_with_interference",                     *args.inputs_with_interference,
            "--regulators_with_interference",                 *args.regulators_with_interference,
            "--interference_evaluation_excluded_node_types",  *args.interference_evaluation_excluded_node_types,
            "--compute_valid_permutations",
            "--percentage",                                   str(args.percentage),
            "--num_epochs_circuit_score_model",               str(args.num_epochs_circuit_score_model),
            "--num_epochs_toxicity_score_model",              str(args.num_epochs_toxicity_score_model),
        ]
        # Only pass --permutation_indices if explicitly restricted; omitting it
        # tells NNGGA to iterate all 6 (paper-default behavior).
        if args.permutation_indices is not None:
            nngga_argv.extend(["--permutation_indices", *args.permutation_indices])
        if not args.no_plotting:
            nngga_argv.append("--plotting")

        # Patch NNGGA's hardcoded DATA_DIR — upstream hardcodes Sebastian's
        # Lincoln Lab Supercloud home dir; on Engaging that path is unreachable
        # and `pd.to_hdf(...)` aborts with no Python-level traceback. We rewrite
        # DATA_DIR to point to tmp_dir so the per-gate_count cache is task-local
        # (see PROJECT_STATE_TOPOLOGY.md §14.9). Only relevant for gate_count<=6;
        # >=7-reg already routes through `--output_dir`.
        HARDCODED_MARKER = (
            "/home/gridsan/spalacios/Designing complex biological circuits with "
            "deep neural networks/dgd/data"
        )
        nngga_source = NNGGA_SCRIPT.read_text()
        if HARDCODED_MARKER not in nngga_source:
            raise SystemExit(
                f"NNGGA source no longer contains expected hardcoded DATA_DIR "
                f"marker '{HARDCODED_MARKER}'. Upstream may have refactored the "
                f"path. Inspect {NNGGA_SCRIPT} and update HARDCODED_MARKER in this "
                f"wrapper. (See PROJECT_STATE_TOPOLOGY.md §14.9.)"
            )
        (tmp_dir / "gate_permutations").mkdir(parents=True, exist_ok=True)
        patched_source = nngga_source.replace(HARDCODED_MARKER, str(tmp_dir))
        patched_script = tmp_dir / "NNGGA_patched.py"
        patched_script.write_text(patched_source)

        print()
        print(f"[run] launching NNGGA via runpy (DATA_DIR rewrite -> {tmp_dir})")
        print(f"[run] argv tail: {' '.join(nngga_argv[1:])[:240]}...")

        sys.argv = nngga_argv
        t0 = time.perf_counter()
        runpy.run_path(str(patched_script), run_name="__main__")
        elapsed = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Promote tmp_dir → final_dir if NNGGA produced something.
    # Counts only *trained* perm_dirs (with both .pt files); interference-
    # rejected perm_dirs (only an h5 file, from NNGGA's eager dir creation
    # in the >=7-reg branch) are expected and ignored.
    # 0-trained case: write marker and promote (meaningful biological
    # finding — see PROJECT_STATE_TOPOLOGY.md §14.10).
    # -------------------------------------------------------------------------
    import json
    all_perm_dirs = sorted(tmp_dir.glob(f"{circuit_name}_design_0_permutation_*"))
    trained_perm_dirs = [d for d in all_perm_dirs if _is_trained_perm_dir(d)]
    n_total = len(all_perm_dirs)
    n_trained = len(trained_perm_dirs)
    n_rejected = n_total - n_trained
    print()
    print(f"[summary] perm_dirs total={n_total}  trained={n_trained}  "
          f"interference-rejected={n_rejected}")

    if n_trained == 0:
        marker = tmp_dir / ZERO_PERMS_MARKER
        marker.write_text(json.dumps({
            "topology_id":     topology_id,
            "circuit_name":    circuit_name,
            "regulator_count": entry.get("regulator_count"),
            "source":          source,
            "energy":          energy,
            "is_baseline":     is_baseline,
            "n_perm_dirs_created": n_total,
            "reason":          "NNGGA returned cleanly with 0 trained perm dirs "
                               "— every surviving input permutation was rejected "
                               "by the truth-table preservation filter "
                               "(NNGGA L162-170) or the interference filter "
                               "(NNGGA L414-416). Topology is unusable under "
                               "default Ara/IPTG interference settings.",
            "filtered_at":     time.time(),
        }, indent=2) + "\n")
        print(f"[zero-perms] {topology_id}: 0 trained perms; wrote {marker.name}")
        print(f"[zero-perms] P5 should tag this topology for L1+ exclusion")

    if final_dir.exists():
        # Should not happen after idempotency check, but be safe
        print(f"[atomicity] final_dir already exists; merging tmp into it")
        for child in tmp_dir.iterdir():
            target = final_dir / child.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(child), str(target))
        shutil.rmtree(tmp_dir)
    else:
        os.rename(tmp_dir, final_dir)

    print()
    print("=" * 64)
    print(f"Phase 4 done for {topology_id}  ({elapsed/60:.1f} min)")
    print(f"  final dir: {final_dir}")
    if (final_dir / ZERO_PERMS_MARKER).exists():
        n_total_final = len(sorted(final_dir.glob(f"{circuit_name}_design_0_permutation_*")))
        print(f"  result   : 0 trained / {n_total_final} created (all filtered) — see {ZERO_PERMS_MARKER}")
    else:
        all_final = sorted(final_dir.glob(f"{circuit_name}_design_0_permutation_*"))
        n_trained_final = sum(1 for d in all_final if _is_trained_perm_dir(d))
        print(f"  result   : {n_trained_final} trained / {len(all_final)} created perm dir(s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
