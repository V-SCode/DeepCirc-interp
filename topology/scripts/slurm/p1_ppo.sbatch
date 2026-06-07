#!/bin/bash
# Phase 1 — PPO+GAT topology generation, one SLURM array task per target.
#
# The array index ($SLURM_ARRAY_TASK_ID, 0-based) selects the i-th target hex
# from configs/target_functions.yaml (whose `id` field is 1-based, so
# target id = SLURM_ARRAY_TASK_ID + 1).
#
# Group invocations:
#   sbatch --array=0-3%1   p1_ppo.sbatch    # G1 (pilot)      — task ids 0..3
#   sbatch --array=4-9%1   p1_ppo.sbatch    # G2 (scale-up)   — task ids 4..9
#   sbatch --array=10-19%1 p1_ppo.sbatch    # G3 (completion) — task ids 10..19
#
# QoS NOTE: the mit_normal_gpu partition's QoS caps each user at 32 CPUs
# concurrent (verified 2026-05-04 via `sacctmgr show qos mit_normal_gpu`).
# We request --cpus-per-task=32 (one full slot per task) and run tasks
# serially (%1), giving upstream-equivalent per-task throughput.
# For more parallelism, switch to --partition=mit_preemptable (longer walltime,
# larger pool, but jobs can be preempted) — see scripts/slurm/README.md.
#
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=p1_ppo
#SBATCH --output=${DEEPCIRC_LOGS:-logs}/p1_%A_%a.out
#SBATCH --error=${DEEPCIRC_LOGS:-logs}/p1_%A_%a.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Activate env explicitly. We do NOT use `source ~/.bashrc; deepcirc` here
# because SLURM compute jobs run a non-interactive shell, and many ~/.bashrc
# files early-exit on non-interactive invocation, leaving the `deepcirc`
# function undefined. Inlining the setup makes the SLURM script self-sufficient.
# -----------------------------------------------------------------------------
module purge
module load miniforge/24.3.0-0

# Conda activation
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/deepcirc"

# libstdc++ fix (CXXABI_1.3.15 needed by IPython/sqlite/icu chain)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Path constants used downstream
export DEEPCIRC_BASE="$HOME/deepcirc"
# TODO: set DEEPCIRC_SCRATCH to your cluster scratch path
export DEEPCIRC_SCRATCH="${DEEPCIRC_SCRATCH:-$HOME/deepcirc_scratch}"

echo "[p1] env activated  conda=$CONDA_PREFIX  python=$(which python)"

TOPOLOGY_DIR="$DEEPCIRC_REPO/topology"  # TODO: set DEEPCIRC_REPO to your DeepCirc-interp checkout
TARGETS_YAML="$TOPOLOGY_DIR/configs/target_functions.yaml"

# -----------------------------------------------------------------------------
# Pick the target hex for this array task.
# Parses the YAML by extracting all `hex:` lines in document order.
# -----------------------------------------------------------------------------
HEXES=( $(awk '/^    hex:/ { gsub(/"/, "", $2); print $2 }' "$TARGETS_YAML") )
N_TARGETS="${#HEXES[@]}"

if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_TARGETS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((N_TARGETS-1)))" >&2
    exit 1
fi

HEX="${HEXES[$SLURM_ARRAY_TASK_ID]}"
echo "[p1] task=$SLURM_ARRAY_TASK_ID  hex=$HEX  job=$SLURM_JOB_ID  array=$SLURM_ARRAY_JOB_ID"

# -----------------------------------------------------------------------------
# Output directory on scratch (large, no backup).
# -----------------------------------------------------------------------------
OUTPUT_DIR="$DEEPCIRC_SCRATCH/runs/stage1/$HEX"
mkdir -p "$OUTPUT_DIR"

# Threading hygiene per upstream's example_zero_shot.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# -----------------------------------------------------------------------------
# Run the wrapper. Hyperparameters match the v2.0 protocol — paper Fig. 2
# benchmarking regime (Palacios et al., Methods L513): 5,120 episodes per
# target, 80 envs, T=10, k=0 initial-state sampling. See pipeline.yaml +
# PROJECT_STATE_TOPOLOGY.md §1.0 / §2 decision #7 / §4.3.
# -----------------------------------------------------------------------------
python "$TOPOLOGY_DIR/scripts/01_generate_topologies.py" \
    --hex "$HEX" \
    --output_dir "$OUTPUT_DIR" \
    --episodes 5120 \
    --n_envs 80 \
    --max_steps 10 \
    --initial_state_sampling_factor 0 \
    --device cuda

echo "[p1] task=$SLURM_ARRAY_TASK_ID hex=$HEX done."
