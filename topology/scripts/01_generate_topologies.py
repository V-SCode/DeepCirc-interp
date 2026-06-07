"""Phase 1 — generate topologies for one target Boolean function.

For a given 3-input target hex (e.g., "0x2B"):

  1. Generate a Verilog file describing the truth table from the hex.
  2. Invoke DeepCirc Stage 1 (PPO+GAT topology agent, zero-shot inference)
     to produce a registry of topology candidates that compute that target.

This script is a thin wrapper around upstream's
`zero_shot_mp_inference_with_registry_main.py`. It applies the
`torch.distributions.set_default_validate_args(False)` fix (required for the
15,929-way action mask softmax under newer torch) and forwards all relevant
hyperparameters from configs/pipeline.yaml.

**v2.0 protocol (locked 2026-05-06): paper-scale zero-shot inference.**
Defaults match the DeepCirc paper's Fig. 2 benchmarking regime
(Palacios et al., Methods lines 503-514): 5,120 episodes (up to 51,200
env-steps), 80 parallel envs, T=10, k=0 initial-state sampling. Earlier
G1 v1.x runs used 1,000 episodes (paper's Fig. 4-5 experimental-
validation regime), which was insufficient for cross-topology
population analysis — hence the v2.0 bump. See PROJECT_STATE_TOPOLOGY.md
§4.3 / §1.0 for the protocol rationale and rerun procedure.

Usage (single-target, manual):
    deepcirc                              # activate env
    python 01_generate_topologies.py \
        --hex 0x2B \
        --output_dir $DEEPCIRC_SCRATCH/runs/stage1/0x2B

Or in a SLURM array job (see scripts/slurm/p1_ppo.sbatch).
"""
from __future__ import annotations

# Apply the torch.distributions fix BEFORE any other imports that touch torch
import torch
torch.distributions.Distribution.set_default_validate_args(False)

import argparse
import os
import runpy
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DEEPCIRC_BASE = Path(os.environ.get("DEEPCIRC_BASE", "~/deepcirc")).expanduser()
UPSTREAM = DEEPCIRC_BASE / "DeepCirc_upstream"
UPSTREAM_MAIN = UPSTREAM / "scripts" / "zero_shot_mp_inference_with_registry_main.py"
MODEL_CHECKPOINT = UPSTREAM / "dgd" / "models" / "trained_agents" / \
    "GAT_MLP_with_scalars_4000_logic_functions" / "trained_model.zip"


# -----------------------------------------------------------------------------
# Verilog generation from hex
# -----------------------------------------------------------------------------
def hex_to_verilog(hex_id: int, module_name: str) -> str:
    """Generate a 3-input Boolean-function Verilog file from a truth-table hex.

    For hex_id `f`, bit `i` of `f` (i ∈ 0..7) is the output for input state `i`,
    where input state `i` decomposes to (C, B, A) = (bit2, bit1, bit0) of `i`.

    Format matches what `dgd.utils.hdl.verilog_to_nig` parses (verified on
    upstream-shipped 0x2C.v during cluster smoke test).
    """
    if not (0 <= hex_id < 256):
        raise ValueError(f"hex_id must be in [0, 256) for 3-input; got {hex_id}")

    bits = [(hex_id >> i) & 1 for i in range(8)]
    lines = [
        f"module {module_name}(A, B, C, Y);",
        "  input  A, B, C;",
        "  output reg Y;",
        "  always @(*) begin",
        "    case ({C, B, A})",
    ]
    for state in range(8):
        lines.append(f"      3'b{state:03b}: Y = 1'b{bits[state]};")
    lines += ["    endcase", "  end", "endmodule", ""]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hex", required=True,
                   help='Target hex, e.g. "0x2B" (case-insensitive)')
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where to write Stage-1 outputs (registry pickle, logs)")

    # Stage-1 hyperparameters — defaults match the DeepCirc paper's Fig. 2
    # benchmarking regime (Methods lines 503-514). See pipeline.yaml.
    p.add_argument("--episodes", type=int, default=5120,
                   help="Paper Fig. 2 zero-shot regime; was 1000 in G1 v1.x "
                        "(Figs. 4-5 experimental-validation regime).")
    p.add_argument("--n_envs", type=int, default=80)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--initial_state_sampling_factor", type=float, default=0.0,
                   help="Paper k parameter for sampling initial topology G_0 "
                        "from the shared registry; weight = 1/C(G)^k. "
                        "k=0 (uniform) is the paper default.")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional reproducibility seed for upstream main")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = p.parse_args()

    # Validate inputs
    hex_str = args.hex.upper().replace("X", "x")  # normalize "0X2B" → "0x2B"
    if not hex_str.startswith("0x"):
        hex_str = "0x" + hex_str
    try:
        hex_int = int(hex_str, 16)
    except ValueError:
        raise SystemExit(f"could not parse --hex {args.hex!r} as hex integer")

    if not UPSTREAM_MAIN.exists():
        raise SystemExit(
            f"upstream main not found at {UPSTREAM_MAIN}. "
            f"Set DEEPCIRC_BASE env var or check repo layout."
        )
    if not MODEL_CHECKPOINT.exists():
        raise SystemExit(f"trained model checkpoint not found at {MODEL_CHECKPOINT}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Phase 1 — topology generation for {hex_str} (decimal {hex_int})")
    print("=" * 64)
    print(f"  output_dir : {args.output_dir}")
    print(f"  upstream   : {UPSTREAM_MAIN}")
    print(f"  model ckpt : {MODEL_CHECKPOINT}")
    print(f"  episodes   : {args.episodes}")
    print(f"  n_envs     : {args.n_envs}")
    print(f"  max_steps  : {args.max_steps}")
    print(f"  init_factor: {args.initial_state_sampling_factor}")

    # Step 1: emit Verilog file
    v_dir = args.output_dir / "seed_verilog"
    v_dir.mkdir(parents=True, exist_ok=True)
    v_path = v_dir / f"{hex_str}.v"
    module_name = hex_str.replace("0x", "")  # Verilog identifier; e.g. "2B"
    v_path.write_text(hex_to_verilog(hex_int, module_name))
    print(f"\n[1] wrote Verilog seed: {v_path} ({v_path.stat().st_size} bytes)")

    # Step 2: forward to upstream zero_shot_mp_inference_with_registry_main.py
    upstream_argv = [
        "zero_shot_mp_inference_with_registry_main",
        "--model_path",                      str(MODEL_CHECKPOINT),
        "--verilog_file",                    str(v_path),
        "--output_folder_name",              str(args.output_dir),
        "--episodes",                        str(args.episodes),
        "--n_envs",                          str(args.n_envs),
        "--max_steps",                       str(args.max_steps),
        "--initial_state_sampling_factor",   str(args.initial_state_sampling_factor),
        "--device",                          args.device,
    ]
    if args.seed is not None:
        upstream_argv.extend(["--seed", str(args.seed)])

    print(f"\n[2] launching upstream PPO topology search:")
    print(f"    {' '.join(upstream_argv[1:])}")
    print()

    sys.argv = upstream_argv
    t0 = time.perf_counter()
    runpy.run_path(str(UPSTREAM_MAIN), run_name="__main__")
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 64)
    print(f"Phase 1 done for {hex_str}  ({elapsed/60:.1f} min)")
    print(f"  Registry: {args.output_dir}/trained_masked/trained_final_shared_registry.pkl")
    print(f"  Metrics : {args.output_dir}/trained_masked/trained_episode_metrics.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
