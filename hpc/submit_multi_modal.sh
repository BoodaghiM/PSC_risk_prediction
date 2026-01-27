#!/bin/bash
#SBATCH --job-name=psc_multi_modal
#SBATCH --time=72:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs_multi_modal/output.out
#SBATCH --error=logs_multi_modal/error.err

set -euo pipefail

# Ensure log dir exists BEFORE SLURM tries to write into it
mkdir -p logs_multi_modal

# --- Work from your repo root ---
REPO_DIR="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "$REPO_DIR"

module purge
module load python

BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"
SCRIPT="modeling/multi_modal.py"

INPUT_DIR="$BASE_DIR/data/data_cleaned/imputed_data"
PHENO_PATH="$BASE_DIR/data/data_cleaned/phenotype_data/data_phenotype_original.csv"

# Output directory (choose ONE approach)
# A) Stable folder (overwrites outputs each run):
# OUT_DIR="$BASE_DIR/results_GitHub/multi_modal"

# B) Timestamped folder (recommended: avoids overwriting):
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$BASE_DIR/results_GitHub/multi_modal"

mkdir -p "$OUT_DIR"

echo "Job ID     : ${SLURM_JOB_ID:-NA}"
echo "Node       : ${SLURMD_NODENAME:-NA}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK:-NA}"
echo "Workdir    : $(pwd)"
echo "Script     : $SCRIPT"
echo "Input dir  : $INPUT_DIR"
echo "Pheno path : $PHENO_PATH"
echo "Out dir    : $OUT_DIR"
echo "Start time : $(date)"

# Prevent BLAS oversubscription when sklearn parallelizes
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --pheno-path "$PHENO_PATH" \
  --out-dir "$OUT_DIR" \
  --seed 42 \
  --n-jobs "${SLURM_CPUS_PER_TASK:-16}" \
  --outer-folds 10 \
  --inner-folds 5 \
  --inner-iters 25 \
  --thr-eval 0.50 \
  --target-recall 0.90 0.85 0.80 \
  --calibration-method sigmoid \
  --calibration-folds 5 \
  --save-calibrators

# To DISABLE calibration, add:
#   --no-calibration
#
# To DISABLE SHAP, add:
#   --no-shap

echo "DONE"
echo "End time: $(date)"

