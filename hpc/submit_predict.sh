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

mkdir -p logs_prediction

INPUT_DIR="/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/imputed_data"
PHENO_PATH="/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/phenotype_data/data_phenotype_original.csv"

OUT="/common/mcgoverndlab/usr/Miad/PSC/results_GitHub/undiagnosed_predictions/scored_undiagnosed_multi_only_raw_and_cal_with_binaries.csv"
mkdir -p "$(dirname "$OUT")"

MULTI_ROOT="/common/mcgoverndlab/usr/Miad/PSC/results_GitHub/multi_modal"
MULTI_MODELS_DIR="${MULTI_ROOT}/models"

THR_CSV="${MULTI_ROOT}/threshold_metrics_oof.csv"
CAL_CLIP="1e-6"

echo "Job ID     : ${SLURM_JOB_ID:-NA}"
echo "Node       : ${SLURMD_NODENAME:-NA}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK:-NA}"
echo "Workdir    : $(pwd)"
echo "Input dir  : ${INPUT_DIR}"
echo "Pheno path : ${PHENO_PATH}"
echo "Out file   : ${OUT}"
echo "Start time : $(date)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CMD=(python evaluation/predict_with_saved_models.py
  --input-dir "${INPUT_DIR}"
  --pheno-path "${PHENO_PATH}"
  --out "${OUT}"
  --undiagnosed-only
  --multi-models-dir "${MULTI_MODELS_DIR}"
)

# If calibrators exist, use them (script will check)
CMD+=(--calibrate-multi --calibrators-dir "${MULTI_MODELS_DIR}" --calibration-clip "${CAL_CLIP}")

# Emit your ORIGINAL-style binary columns (script will filter to multi only)
if [[ -f "${THR_CSV}" ]]; then
  CMD+=(--emit-multi-binaries --thresholds-csv "${THR_CSV}")
else
  echo "WARNING: thresholds CSV not found: ${THR_CSV} (no binaries emitted)"
fi

echo "Running command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Done: $(date)"

