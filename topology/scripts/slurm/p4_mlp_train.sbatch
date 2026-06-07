#!/bin/bash
# Phase 4 — per-topology Stage-2 MLP training (one SLURM array task per topology).
#
# The wrapper (scripts/04_train_mlps.py) is idempotent and uses atomic dir-rename
# semantics, so this template works on either:
#   - mit_normal_gpu (slow, polite, no preemption) — for G1
#   - mit_preemptable (faster, more concurrency, may preempt; --requeue rescues)
#
# G1 invocations (~92 topologies):
#   sbatch --array=0-91%1 p4_mlp_train.sbatch                              # serial on mit_normal_gpu
#
# G2 / G3 invocations (later, with preemption-tolerant setup):
#   sbatch --array=0-N%8 --partition=mit_preemptable --requeue p4_mlp_train.sbatch
#
# Override the population manifest path via env var:
#   POPULATION_MANIFEST=$DEEPCIRC_SCRATCH/population/G2/population.pkl sbatch ...
#
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=p4_mlp
#SBATCH --output=${DEEPCIRC_LOGS:-logs}/p4_%A_%a.out
#SBATCH --error=${DEEPCIRC_LOGS:-logs}/p4_%A_%a.err

set -euo pipefail

# Inline env activation
module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/deepcirc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DEEPCIRC_BASE="$HOME/deepcirc"
# TODO: set DEEPCIRC_SCRATCH to your cluster scratch path
export DEEPCIRC_SCRATCH="${DEEPCIRC_SCRATCH:-$HOME/deepcirc_scratch}"
echo "[p4] env activated  conda=$CONDA_PREFIX  python=$(which python)"

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

echo "[p4] task=$SLURM_ARRAY_TASK_ID  group=$GROUP"
echo "[p4] manifest=$POPULATION_MANIFEST"
echo "[p4] output_root=$OUTPUT_ROOT"

python "$TOPOLOGY_DIR/scripts/04_train_mlps.py" \
    --population_manifest "$POPULATION_MANIFEST" \
    --topology_index "$SLURM_ARRAY_TASK_ID" \
    --output_root "$OUTPUT_ROOT"

echo "[p4] task=$SLURM_ARRAY_TASK_ID done."
