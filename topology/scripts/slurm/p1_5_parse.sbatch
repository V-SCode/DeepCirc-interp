#!/bin/bash
# Phase 1.5 — registry parsing (CPU-only, fast).
#
# One SLURM array task per target hex. Invokes upstream's
# `shared_registry_parsing_HDL_main.py` to produce `optimal_topologies.pkl`
# per target.
#
# Group invocations:
#   sbatch --array=0-3%4    p1_5_parse.sbatch   # G1
#   sbatch --array=4-9%6    p1_5_parse.sbatch   # G2
#   sbatch --array=10-19%6  p1_5_parse.sbatch   # G3
#
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=p1_5_parse
#SBATCH --output=${DEEPCIRC_LOGS:-logs}/p1_5_%A_%a.out
#SBATCH --error=${DEEPCIRC_LOGS:-logs}/p1_5_%A_%a.err

set -euo pipefail

# Inline env activation (same pattern as p1_ppo.sbatch — avoids /etc/bashrc)
module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/deepcirc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DEEPCIRC_BASE="$HOME/deepcirc"
# TODO: set DEEPCIRC_SCRATCH to your cluster scratch path
export DEEPCIRC_SCRATCH="${DEEPCIRC_SCRATCH:-$HOME/deepcirc_scratch}"
echo "[p1.5] env activated  conda=$CONDA_PREFIX  python=$(which python)"

TOPOLOGY_DIR="$DEEPCIRC_REPO/topology"  # TODO: set DEEPCIRC_REPO to your DeepCirc-interp checkout
TARGETS_YAML="$TOPOLOGY_DIR/configs/target_functions.yaml"

HEXES=( $(awk '/^    hex:/ { gsub(/"/, "", $2); print $2 }' "$TARGETS_YAML") )
HEX="${HEXES[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$HEX" ]; then
    echo "ERROR: no hex for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

RUN_DIR="$DEEPCIRC_SCRATCH/runs/stage1/$HEX/trained_masked"
echo "[p1.5] task=$SLURM_ARRAY_TASK_ID hex=$HEX run_dir=$RUN_DIR"

python "$TOPOLOGY_DIR/scripts/01_5_parse_registries.py" \
    --hex "$HEX" \
    --run_dir "$RUN_DIR" \
    --mode zero-shot

echo "[p1.5] task=$SLURM_ARRAY_TASK_ID done."
