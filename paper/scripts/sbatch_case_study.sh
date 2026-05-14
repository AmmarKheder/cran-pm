#!/bin/bash -l
#SBATCH --job-name=cranpm-case-study
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/case_%x_%j.out
#SBATCH --error=logs/case_%x_%j.err

# CRAN-PM GMD paper — case-study inference + figure generation.
# Usage:
#   sbatch sbatch_case_study.sh saharan-dust 2022-03-14 2022-03-19
#   sbatch sbatch_case_study.sh iberian-fires 2022-07-14 2022-07-22
#   sbatch sbatch_case_study.sh polish-winter 2022-01-08 2022-01-18

CASE_NAME="${1:?missing case name}"
DATE_START="${2:?missing start date YYYY-MM-DD}"
DATE_END="${3:?missing end date YYYY-MM-DD}"

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-db
mkdir -p "$MIOPEN_USER_DB_PATH"

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

CHECKPOINT=/pfs/lustrep1/scratch/project_462001140/ammar/eccv/topoflow_europe/paper_checkpoints/cranpm_v3.ckpt
OUT_DIR=/scratch/project_462001140/ammar/eccv/cran-pm/paper/figures_cases/${CASE_NAME}
mkdir -p "$OUT_DIR"

cd /scratch/project_462001140/ammar/eccv/cran-pm

srun python paper/scripts/run_case_study.py \
    --checkpoint "$CHECKPOINT" \
    --date-start "$DATE_START" \
    --date-end "$DATE_END" \
    --lead-times 1 \
    --precision bf16 \
    --output-zarr "$OUT_DIR/predictions.zarr"

# Generate paper figures from the produced zarr.
srun python /scratch/project_462001140/ammar/eccv/cran-pm/paper/scripts/fig_case_study.py \
    --case "$CASE_NAME" \
    --predictions "$OUT_DIR/predictions.zarr" \
    --date-start "$DATE_START" \
    --date-end "$DATE_END" \
    --output-dir "$OUT_DIR"

echo "Done: $CASE_NAME ($DATE_START -- $DATE_END)"
