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

# --- Work from repo root ---
REPO_ROOT="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "$REPO_ROOT"

# ensure log dir exists BEFORE SLURM tries to write into it
mkdir -p logs_single_modal

# -------------------------
# Paths
# -------------------------
BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"
INPUT_DIR="${BASE_DIR}/data/data_cleaned/imputed_data"
PHENO_PATH="${BASE_DIR}/data/data_cleaned/phenotype_data/data_phenotype_original.csv"

# Where to save outputs
OUT_ROOT="${BASE_DIR}/results_GitHub/single_modal"

# Script location inside repo
SCRIPT="modeling/single_modal.py"

# -------------------------
# Map array index -> modality
# -------------------------
MODALITIES=(genetics lab serology clinical)
MODALITY="${MODALITIES[$SLURM_ARRAY_TASK_ID]}"

# NEW run folder to avoid overwriting old results
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/${MODALITY}"
mkdir -p "${OUT_DIR}"

echo "Job ID        : ${SLURM_JOB_ID:-NA}"
echo "Array task    : ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node          : ${SLURMD_NODENAME:-NA}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-NA}"
echo "Workdir       : $(pwd)"
echo "Modality      : ${MODALITY}"
echo "Input dir     : ${INPUT_DIR}"
echo "Pheno path    : ${PHENO_PATH}"
echo "Out dir       : ${OUT_DIR}"
echo "Start time    : $(date)"

# Repro / avoid oversubscription
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "${SCRIPT}" \
  --modality "${MODALITY}" \
  --input-dir "${INPUT_DIR}" \
  --pheno-path "${PHENO_PATH}" \
  --out-dir "${OUT_DIR}" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-16}" \
  --seed 42 \
  --outer-folds 10 \
  --inner-folds 5 \
  --inner-iters 25 \
  --thr-eval 0.50 \
  --target-recall 0.90 0.85 0.80 \
  --calibration-method sigmoid \
  --calibration-folds 5 \
  --save-calibrators \
  --calibration-clip 1e-6

# If you want to DISABLE calibration, replace the calibration lines with:
#   --no-calibration
#
# If you want to DISABLE SHAP (faster/less memory), add:
#   --no-shap
#
# If you don't want to save the deployable OOF probability-mapper calibrators, remove:
#   --save-calibrators

echo "DONE [${MODALITY}]"
echo "End time: $(date)"

