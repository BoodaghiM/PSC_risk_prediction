#!/bin/bash
#SBATCH --job-name=psc_rf_single
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3
#SBATCH --output=logs_single_modal/%x_%A_%a.out
#SBATCH --error=logs_single_modal/%x_%A_%a.err

set -euo pipefail

module purge
module load python

REPO_ROOT="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "$REPO_ROOT"

mkdir -p logs_single_modal

# -------------------------
# Paths (edit if needed)
# -------------------------
BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"
INPUT_DIR="${BASE_DIR}/data/data_cleaned/imputed_data"
PHENO_PATH="${BASE_DIR}/data/data_cleaned/phenotype_data/data_phenotype_original.csv"

# Where to save single-modality outputs (outside repo is fine)
OUT_ROOT="${BASE_DIR}/results_GitHub/single_modal"

# Script location inside your repo
SCRIPT="modeling/single_modal.py"

# -------------------------
# Map array index -> modality
# -------------------------
MODALITIES=(genetics lab serology clinical)
MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

OUT_DIR="${OUT_ROOT}/${MODALITY}"
mkdir -p "${OUT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Running modality: ${MODALITY}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Input dir: ${INPUT_DIR}"
echo "Pheno: ${PHENO_PATH}"
echo "Out dir: ${OUT_DIR}"
echo "Start time: $(date)"

python "${SCRIPT}" \
  --modality "${MODALITY}" \
  --input-dir "${INPUT_DIR}" \
  --pheno-path "${PHENO_PATH}" \
  --out-dir "${OUT_DIR}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 42

echo "DONE [${MODALITY}]"
echo "End time: $(date)"

