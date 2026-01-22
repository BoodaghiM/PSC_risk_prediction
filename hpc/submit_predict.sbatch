#!/bin/bash
#SBATCH --job-name=psc_score_undiagnosed
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs_prediction/%x_%j.out
#SBATCH --error=logs_prediction/%x_%j.err

set -euo pipefail

REPO_DIR="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "$REPO_DIR" || exit 1

module purge
module load python

# Create logs dir INSIDE repo under hpc/
mkdir -p logs_prediction

# ---- Paths ----
INPUT_DIR="/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/imputed_data/"
PHENO_PATH="/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/phenotype_data/data_phenotype_original.csv"
OUT="/common/mcgoverndlab/usr/Miad/PSC/results_GitHub/undiagnosed_predictions/scored_undiagnosed.csv"
mkdir -p "$(dirname "$OUT")"

# ---- Model directories ----
SINGLE_ROOT="/common/mcgoverndlab/usr/Miad/PSC/results_GitHub/single_modal"

# Pick latest multi-modal run automatically
MULTI_ROOT="/common/mcgoverndlab/usr/Miad/PSC/results_GitHub/multi_modal"
LATEST_RUN="$(ls -1dt ${MULTI_ROOT}/run_* 2>/dev/null | head -n 1)"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "ERROR: No run_* directories found under ${MULTI_ROOT}"
  exit 1
fi
MULTI_MODELS_DIR="${LATEST_RUN}/models"

LAB_MODEL_DIR="${SINGLE_ROOT}/lab"
SEROLOGY_MODEL_DIR="${SINGLE_ROOT}/serology"
CLINICAL_MODEL_DIR="${SINGLE_ROOT}/clinical"
GENETICS_MODEL_DIR="${SINGLE_ROOT}/genetics"


python evaluation/predict_with_saved_models.py \
  --input-dir "$INPUT_DIR" \
  --pheno-path "$PHENO_PATH" \
  --out "$OUT" \
  --undiagnosed-only \
  --lab-model-dir "$LAB_MODEL_DIR" \
  --serology-model-dir "$SEROLOGY_MODEL_DIR" \
  --clinical-model-dir "$CLINICAL_MODEL_DIR" \
  --genetics-model-dir "$GENETICS_MODEL_DIR" \
  --multi-models-dir "$MULTI_MODELS_DIR" \
  --compute-thresholds

echo "Done: $(date)"


