#!/bin/bash
#SBATCH --job-name=psc_multi_modal
#SBATCH --time=72:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16

# log files
#SBATCH --output=logs_multi_modal/%x_%j.out
#SBATCH --error=logs_multi_modal/%x_%j.err

set -euo pipefail

# --- Work from your repo root ---
REPO_DIR="/common/mcgoverndlab/usr/Miad/PSC/PSC_GitHub"
cd "$REPO_DIR"

module purge
module load python

# ensure log dir exists on HPC
mkdir -p logs_multi_modal

BASE_DIR="/common/mcgoverndlab/usr/Miad/PSC"
SCRIPT="modeling/multi_modal.py"

INPUT_DIR="$BASE_DIR/data/data_cleaned/imputed_data"
PHENO_PATH="$BASE_DIR/data/data_cleaned/phenotype_data/data_phenotype_original.csv"

# IMPORTANT: choose a NEW run folder so you don’t overwrite old models/results
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$BASE_DIR/results_GitHub/multi_modal/run_${TS}"
mkdir -p "$OUT_DIR"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Workdir    : $(pwd)"
echo "Script     : $SCRIPT"
echo "Out dir    : $OUT_DIR"
echo "Start time : $(date)"

python "$SCRIPT" \
  --input-dir "$INPUT_DIR" \
  --pheno-path "$PHENO_PATH" \
  --out-dir "$OUT_DIR" \
  --seed 42 \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --outer-folds 10 \
  --inner-folds 5 \
  --inner-iters 25

echo "DONE"
echo "End time: $(date)"

