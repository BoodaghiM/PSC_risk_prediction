#!/bin/bash
#SBATCH --job-name=psc_risk_assoc
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs_risk_prediction_associations/%x_%j.out
#SBATCH --error=logs_risk_prediction_associations/%x_%j.err

set -euo pipefail

module purge
source /home/boodaghidim/miniconda/etc/profile.d/conda.sh
conda activate base

# --- Work from repo root ---
REPO_ROOT="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "${REPO_ROOT}"

# make sure this exists before sbatch in practice
mkdir -p logs_risk_prediction_associations

# -------------------------
# Paths
# -------------------------
INPUT_FILE="/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/adjudication/PSC_blind_validation_merged_with_scores_and_discordance.xlsx"
OUT_DIR="/common/mcgoverndlab/usr/Miad/PSC/results_GitHub/risk_prediction_associations"

# Script location inside repo
SCRIPT="evaluation/categorical_risk_analysis.py"

mkdir -p "${OUT_DIR}"

echo "Job ID        : ${SLURM_JOB_ID:-NA}"
echo "Node          : ${SLURMD_NODENAME:-NA}"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-NA}"
echo "Workdir       : $(pwd)"
echo "Input file    : ${INPUT_FILE}"
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
  --input "${INPUT_FILE}" \
  --sheet 0 \
  --outdir "${OUT_DIR}" \
  --psc-col PSC \
  --risk-col risk_multi_late \
  --fibrosis-col "Fibrosis stage" \
  --fig-width 6.5 \
  --fig-height 5.5 \
  --dpi 1000

echo "DONE"
echo "End time: $(date)"
