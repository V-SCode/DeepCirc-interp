#!/bin/bash
# Phase 4 — per-topology Stage-2 MLP training on mit_preemptable.
#
# Identical wrapper invocation to p4_mlp_train.sbatch, but uses the much-larger
# mit_preemptable partition (per-user caps verified 2026-05-04: 1024 CPUs,
# 4 GPUs, 4 TB memory). With --requeue + the wrapper's idempotency + atomic
# dir-rename semantics, preempted tasks restart cleanly from scratch on
# their next array slot — no manual recovery needed.
#
# Concurrency: --array=...%4 to hit the 4-GPU cap (any higher will queue).
# Walltime: 12 h (well within the 48 h preemptable cap; per-topology runs are
# typically 7–22 min, so 12 h covers the rare slow case + restart noise).
#
# Group invocations:
#   GROUP=G1 sbatch --array=0-91%4   p4_mlp_train_preemptable.sbatch
#   GROUP=G2 sbatch --array=0-249%4  p4_mlp_train_preemptable.sbatch
#   GROUP=G3 sbatch --array=0-499%4  p4_mlp_train_preemptable.sbatch
#
# (population size N for each group can be read from population_summary.json:
#   N=$(jq .n_total $DEEPCIRC_SCRATCH/population/$GROUP/population_summary.json)
#   sbatch --array=0-$((N-1))%4 p4_mlp_train_preemptable.sbatch)
#
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --job-name=p4_mlp_pe
#SBATCH --output=${DEEPCIRC_LOGS:-logs}/p4pe_%A_%a.out
#SBATCH --error=${DEEPCIRC_LOGS:-logs}/p4pe_%A_%a.err

set -euo pipefail

# Inline env activation (avoids /etc/bashrc strict-mode issue per §14.2)
module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/deepcirc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DEEPCIRC_BASE="$HOME/deepcirc"
# TODO: set DEEPCIRC_SCRATCH to your cluster scratch path
export DEEPCIRC_SCRATCH="${DEEPCIRC_SCRATCH:-$HOME/deepcirc_scratch}"
echo "[p4-pe] env activated  conda=$CONDA_PREFIX  python=$(which python)"
echo "[p4-pe] partition=mit_preemptable  array=$SLURM_ARRAY_JOB_ID"
echo "[p4-pe] task=$SLURM_ARRAY_TASK_ID  attempt=${SLURM_RESTART_COUNT:-0}"

# Threading hygiene per upstream
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

TOPOLOGY_DIR="$DEEPCIRC_REPO/topology"  # TODO: set DEEPCIRC_REPO to your DeepCirc-interp checkout
GROUP="${GROUP:-G1}"
POPULATION_MANIFEST="${POPULATION_MANIFEST:-$DEEPCIRC_SCRATCH/population/$GROUP/population.pkl}"
# OUTPUT_ROOT — group-agnostic per-topology pool (Option C, R5).
# Trained MLPs live under runs/stage2/<topology_id>/ regardless of group, so
# G2/G3 P4 fires reuse G1's already-trained topologies via the wrapper's
# idempotent already-trained check.
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEEPCIRC_SCRATCH/runs/stage2}"

if [ ! -f "$POPULATION_MANIFEST" ]; then
    echo "ERROR: population manifest not found: $POPULATION_MANIFEST" >&2
    echo "       Run Phase 3 first (assemble_population.py) for group $GROUP" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "[p4-pe] task=$SLURM_ARRAY_TASK_ID  group=$GROUP"
echo "[p4-pe] manifest=$POPULATION_MANIFEST"
echo "[p4-pe] output_root=$OUTPUT_ROOT"

python "$TOPOLOGY_DIR/scripts/04_train_mlps.py" \
    --population_manifest "$POPULATION_MANIFEST" \
    --topology_index "$SLURM_ARRAY_TASK_ID" \
    --output_root "$OUTPUT_ROOT"

echo "[p4-pe] task=$SLURM_ARRAY_TASK_ID done."
