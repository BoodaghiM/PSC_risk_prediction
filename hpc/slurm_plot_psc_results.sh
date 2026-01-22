#!/bin/bash
#SBATCH --job-name=psc_plots
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs_plots/%x_%j.out
#SBATCH --error=logs_plots/%x_%j.err

set -euo pipefail

module purge
module load python

# -------------------------
# Paths (edit if needed)
# -------------------------
REPO_ROOT="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"

# Where your results live
RESULTS_ROOT="${BASE_DIR}/results_GitHub"
RUN_DIR="${RESULTS_ROOT}/multi_modal"                  # contains oof_predictions.csv, shap_multi_early.csv, shap_multi_late_meta_lr.csv
OUT_DIR="${RESULTS_ROOT}/plots"                        # plots will be written here

# Old single-modality nested CV OOFs (optional)
SINGLE_ROOT="${RESULTS_ROOT}/single_modal"
OLD_CLINICAL="${SINGLE_ROOT}/clinical/nestedcv_oof_predictions.csv"
OLD_GENETICS="${SINGLE_ROOT}/genetics/nestedcv_oof_predictions.csv"
OLD_LAB="${SINGLE_ROOT}/lab/nestedcv_oof_predictions.csv"
OLD_SEROLOGY="${SINGLE_ROOT}/serology/nestedcv_oof_predictions.csv"

# Script location inside repo
SCRIPT="evaluation/plot_psc_results.py"

mkdir -p "${OUT_DIR}"
mkdir -p logs_plots

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Run dir: ${RUN_DIR}"
echo "Out dir: ${OUT_DIR}"
echo "Start time: $(date)"

# -------------------------
# Run
# -------------------------
python "${SCRIPT}" \
  --run-dir "${RUN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --dpi 300 \
  --also-pdf \
  --top-k 20 \
  --single-clinical "${OLD_CLINICAL}" \
  --single-genetics "${OLD_GENETICS}" \
  --single-lab "${OLD_LAB}" \
  --single-serology "${OLD_SEROLOGY}"

echo "DONE"
echo "End time: $(date)"

