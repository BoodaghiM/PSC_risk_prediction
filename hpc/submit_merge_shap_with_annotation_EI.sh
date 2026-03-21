#!/bin/bash
#SBATCH --job-name=merge_shap_annot
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs_postprocessing/%x_%A.out
#SBATCH --error=logs_postprocessing/%x_%A.err

set -euo pipefail

module purge
module load python

# --- Work from repo root ---
REPO_ROOT="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "$REPO_ROOT"

# ensure log dir exists
mkdir -p logs_postprocessing

# -------------------------
# Paths
# -------------------------
BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"

ANNOTATION_PATH="${BASE_DIR}/data/data_cleaned/mapping_data/MIRIAD_annotated.hg38_multianno.txt"
SHAP_PATH="${BASE_DIR}/results_GitHub/multi_modal/shap_multi_early_aggregated.csv"

# Output
OUT_DIR="${BASE_DIR}/results_GitHub/multi_modal"
OUT_FILE="${OUT_DIR}/shap_multi_early_annotated.csv"
mkdir -p "${OUT_DIR}"

# Script location inside repo
SCRIPT="post_processing/merge_shap_with_annotation.py"

echo "Job ID        : ${SLURM_JOB_ID:-NA}"
echo "Node          : ${SLURMD_NODENAME:-NA}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-NA}"
echo "Workdir       : $(pwd)"
echo "Annotation    : ${ANNOTATION_PATH}"
echo "SHAP input    : ${SHAP_PATH}"
echo "Output file   : ${OUT_FILE}"
echo "Start time    : $(date)"

# Repro / avoid oversubscription
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "${SCRIPT}" \
  --annotation "${ANNOTATION_PATH}" \
  --shap "${SHAP_PATH}" \
  --output "${OUT_FILE}" \
  --round-decimals 4 \
  --output-format csv

echo "DONE"
echo "End time: $(date)"
