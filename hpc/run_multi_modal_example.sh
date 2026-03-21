#!/bin/bash
#SBATCH --job-name=psc_multi_modal
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load python

# -------------------------
# Edit these paths
# -------------------------
REPO_ROOT="/path/to/PSC_risk_prediction"
DATA_ROOT="/path/to/data"
OUT_DIR="/path/to/results/multi_modal"

INPUT_DIR="${DATA_ROOT}/imputed_data"
PHENO_PATH="${DATA_ROOT}/phenotype_data/data_phenotype_original.csv"

mkdir -p "${OUT_DIR}"
mkdir -p logs

cd "${REPO_ROOT}"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python modeling/multi_modal.py \
  --input-dir "${INPUT_DIR}" \
  --pheno-path "${PHENO_PATH}" \
  --out-dir "${OUT_DIR}" \
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
