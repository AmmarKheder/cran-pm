#!/bin/bash -l
#SBATCH --job-name=cranpm-zs-india-test
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/zs_test_india_%j.out
#SBATCH --error=logs/zs_test_india_%j.err

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-db
mkdir -p "$MIOPEN_USER_DB_PATH"

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

export CRANPM_OOD_CKPT=/scratch/project_462001140/ammar/eccv/topoflow_europe/checkpoints_v10f/topoflow-018.ckpt
export CRANPM_OOD_EVAL_DIR=/scratch/project_462001140/ammar/eccv/topoflow_europe/evaluation_ood_v10f_otf

cd /scratch/project_462001140/ammar/eccv/topoflow_europe

# Test one region first
srun python scripts/run_inference_ood_regions.py india

echo "Done. Results in $CRANPM_OOD_EVAL_DIR/india"
ls -la "$CRANPM_OOD_EVAL_DIR/india" 2>/dev/null
cat "$CRANPM_OOD_EVAL_DIR/india/metrics.json" 2>/dev/null
