"""Phase 1.5 — wrapper around upstream's `shared_registry_parsing_HDL_main.py`.

For one target hex (e.g., "0x2B"), invoke upstream's registry parser on the
corresponding Stage-1 output directory. The parser produces:

    <run_dir>/optimal_topologies/optimal_topologies.pkl    (list[networkx.DiGraph])

This is the paper-faithful Pareto-optimal subset that NNGGA expects as input.

Why this phase exists: our Phase 3 was originally taking raw registries directly
and running our own (energy-only) Pareto. Per PROJECT_STATE_TOPOLOGY.md §2.1
"match upstream defaults," we route through upstream's parser so our cross-
topology MLPs are trained on exactly the same Pareto-filtered designs the
DeepCirc paper trained on.

Usage:
    python 01_5_parse_registries.py --hex 0x2B \\
        --run_dir $DEEPCIRC_SCRATCH/runs/stage1/0x2B/trained_masked

Or in a SLURM array job (see scripts/slurm/p1_5_parse.sbatch).
"""
from __future__ import annotations

# Apply torch.distributions fix in case anything in the upstream parser path
# hits torch (it shouldn't, but defensive).
import torch
torch.distributions.Distribution.set_default_validate_args(False)

import argparse
import os
import runpy
import sys
import time
from pathlib import Path

DEEPCIRC_BASE = Path(os.environ.get("DEEPCIRC_BASE", "~/deepcirc")).expanduser()
UPSTREAM = DEEPCIRC_BASE / "DeepCirc_upstream"
UPSTREAM_PARSER = UPSTREAM / "scripts" / "shared_registry_parsing_HDL_main.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hex", required=True, help='Target hex, e.g. "0x2B"')
    p.add_argument("--run_dir", required=True, type=Path,
                   help="Path to <stage1_output>/<HEX>/trained_masked/ "
                        "(contains trained_final_shared_registry.pkl)")
    p.add_argument("--mode", default="zero-shot",
                   choices=["scratch", "fine-tune", "zero-shot"],
                   help="Which run mode produced the registry (matches Stage-1 mode)")
    args = p.parse_args()

    if not UPSTREAM_PARSER.exists():
        raise SystemExit(f"upstream parser not found at {UPSTREAM_PARSER}")
    if not args.run_dir.exists():
        raise SystemExit(f"run_dir not found: {args.run_dir}")

    expected_registry = args.run_dir / "trained_final_shared_registry.pkl"
    if not expected_registry.exists():
        raise SystemExit(f"expected registry not found at {expected_registry}")

    out_pkl = args.run_dir / "optimal_topologies" / "optimal_topologies.pkl"
    if out_pkl.exists():
        print(f"[1.5] {args.hex}: optimal_topologies.pkl already exists at {out_pkl} — skipping")
        return

    print("=" * 64)
    print(f"Phase 1.5 — registry parsing for {args.hex}")
    print("=" * 64)
    print(f"  run_dir      : {args.run_dir}")
    print(f"  registry     : {expected_registry} ({expected_registry.stat().st_size} bytes)")
    print(f"  parser       : {UPSTREAM_PARSER}")
    print(f"  mode         : {args.mode}")
    print()

    # Forward-call the upstream parser via runpy
    upstream_argv = [
        "shared_registry_parsing_HDL_main",
        "--run_dir", str(args.run_dir),
        "--hex_ID", args.hex,
        "--mode", args.mode,
    ]
    sys.argv = upstream_argv

    t0 = time.perf_counter()
    runpy.run_path(str(UPSTREAM_PARSER), run_name="__main__")
    elapsed = time.perf_counter() - t0

    print()
    if out_pkl.exists():
        print("=" * 64)
        print(f"Phase 1.5 done for {args.hex}  ({elapsed:.1f} s)")
        print(f"  → {out_pkl}  ({out_pkl.stat().st_size} bytes)")
        print("=" * 64)
    else:
        raise SystemExit(f"parser claimed success but {out_pkl} was not produced")


if __name__ == "__main__":
    main()
