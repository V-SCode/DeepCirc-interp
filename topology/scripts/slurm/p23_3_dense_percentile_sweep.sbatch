#!/bin/bash
# Phase 23.3 — dense per-size percentile sweep for fig31_v3 (single-node, CPU).
#
# Submit with:
#   sbatch topology/scripts/slurm/p23_3_dense_percentile_sweep.sbatch
#
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=p23_3
#SBATCH --output=${DEEPCIRC_LOGS:-logs}/p23_3_%j.out
#SBATCH --error=${DEEPCIRC_LOGS:-logs}/p23_3_%j.err

set -euo pipefail

module purge
module load miniforge/24.3.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/deepcirc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export DEEPCIRC_BASE="$HOME/deepcirc"
# TODO: set DEEPCIRC_SCRATCH to your cluster scratch path
export DEEPCIRC_SCRATCH="${DEEPCIRC_SCRATCH:-$HOME/deepcirc_scratch}"
echo "[p23_3] env activated  conda=$CONDA_PREFIX  python=$(which python)"
mkdir -p $DEEPCIRC_SCRATCH/logs

TOPOLOGY_DIR="$DEEPCIRC_REPO/topology"  # TODO: set DEEPCIRC_REPO to your DeepCirc-interp checkout
OUTPUT_ROOT="$DEEPCIRC_SCRATCH/population/G3/size_tier_designs"
mkdir -p "$OUTPUT_ROOT"

# Default invocation: all targets, no suffix.
python "$TOPOLOGY_DIR/scripts/23_3_dense_percentile_sweep.py" \
    --population_manifest "$DEEPCIRC_SCRATCH/population/G3/population.pkl" \
    --qc_tiers            "$DEEPCIRC_SCRATCH/population/G3/qc_tiers.pkl" \
    --predictions_root    "$DEEPCIRC_SCRATCH/topology_data/design_space_predictions" \
    --output_root         "$OUTPUT_ROOT"

# 3-input fully-functional pool (drops 1-input degenerate NPN 0x0F: 0x33/0xCC/0x0F
# and 2-input degenerate NPN 0x03: 0xEE/0x03/0x77 and NPN 0x3C: 0x66/0x3C).
# Outputs dense_percentile_sweep_3input.json.
python "$TOPOLOGY_DIR/scripts/23_3_dense_percentile_sweep.py" \
    --population_manifest "$DEEPCIRC_SCRATCH/population/G3/population.pkl" \
    --qc_tiers            "$DEEPCIRC_SCRATCH/population/G3/qc_tiers.pkl" \
    --predictions_root    "$DEEPCIRC_SCRATCH/topology_data/design_space_predictions" \
    --output_root         "$OUTPUT_ROOT" \
    --include_targets 0x17 0x2B 0x4D 0xE8 0x71 0x8E 0x6D 0x47 0x1B 0x06 0x07 0x2C \
    --output_suffix _3input

echo "[p23_3] done"
