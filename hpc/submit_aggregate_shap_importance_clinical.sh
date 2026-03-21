#!/bin/bash
#SBATCH --job-name=agg_shap_clinical
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs_postprocessing/%x_%A.out
#SBATCH --error=logs_postprocessing/%x_%A.err

set -euo pipefail

module purge
module load python

# -------------------------
# Work from repo root
# -------------------------
REPO_ROOT="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "${REPO_ROOT}"

# ensure log dir exists
mkdir -p logs_postprocessing

# -------------------------
# Paths
# -------------------------
BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"

INPUT_FILE="${BASE_DIR}/results_GitHub/single_modal/clinical/shap_global_importance.csv"
OUTPUT_FILE="${BASE_DIR}/results_GitHub/single_modal/clinical/shap_global_importance_aggregated.csv"

SCRIPT="post_processing/aggregate_shap_importance.py"

echo "Job ID        : ${SLURM_JOB_ID:-NA}"
echo "Node          : ${SLURMD_NODENAME:-NA}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-NA}"
echo "Workdir       : $(pwd)"
echo "Input file    : ${INPUT_FILE}"
echo "Output file   : ${OUTPUT_FILE}"
echo "Script        : ${SCRIPT}"
echo "Start time    : $(date)"

# avoid oversubscription
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "${SCRIPT}" \
  --input "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --round-decimals 4

echo "DONE"
echo "End time: $(date)"
