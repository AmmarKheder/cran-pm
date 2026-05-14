#!/bin/bash -l
#SBATCH --job-name=cranpm-zero-shot-ood
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=logs/zero_shot_%j.out
#SBATCH --error=logs/zero_shot_%j.err

# Zero-shot OOD inference on 5 global regions using the v3 (GMD headline)
# checkpoint. One GPU per region, runs in parallel.
# Outputs go to evaluation_ood_v10f_otf/<region>/{predictions_t1.zarr,metrics.json}

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-db
mkdir -p "$MIOPEN_USER_DB_PATH"

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

export CRANPM_OOD_CKPT=/scratch/project_462001140/ammar/eccv/topoflow_europe/checkpoints_v10f/topoflow-018.ckpt
export CRANPM_OOD_EVAL_DIR=/scratch/project_462001140/ammar/eccv/topoflow_europe/evaluation_ood_v10f_otf

cd /scratch/project_462001140/ammar/eccv/topoflow_europe

srun python scripts/run_inference_ood_regions.py all

echo "Zero-shot OOD done. Results in $CRANPM_OOD_EVAL_DIR"
ls -la "$CRANPM_OOD_EVAL_DIR"
