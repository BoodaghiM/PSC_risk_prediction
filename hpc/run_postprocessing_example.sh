#!/bin/bash
#SBATCH --job-name=psc_postprocess
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load python

# -------------------------
# Edit these paths
# -------------------------
REPO_ROOT="/path/to/PSC_risk_prediction"
RESULTS_ROOT="/path/to/results"
ANNOTATION_PATH="/path/to/MIRIAD_annotated.hg38_multianno.txt"

mkdir -p logs
cd "${REPO_ROOT}"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Example 1: aggregate SHAP importance
python post_processing/aggregate_shap_importance.py \
  --input "${RESULTS_ROOT}/single_modal/clinical/shap_global_importance.csv" \
  --output "${RESULTS_ROOT}/single_modal/clinical/shap_global_importance_aggregated.csv" \
  --round-decimals 4

# Example 2: merge genetics SHAP with annotation
python post_processing/merge_shap_with_annotation.py \
  --annotation "${ANNOTATION_PATH}" \
  --shap "${RESULTS_ROOT}/single_modal/genetics/shap_global_importance.csv" \
  --output "${RESULTS_ROOT}/single_modal/genetics/shap_global_importance_annotated.csv" \
  --round-decimals 4 \
  --output-format csv

# Example 3: create evaluation plots
python evaluation/plot_psc_results.py \
  --run-dir "${RESULTS_ROOT}/multi_modal" \
  --out-dir "${RESULTS_ROOT}/plots" \
  --dpi 300 \
  --top-k 20 \
  --calib-bins 10
