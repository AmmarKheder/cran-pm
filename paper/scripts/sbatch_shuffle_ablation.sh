#!/bin/bash -l
#SBATCH --job-name=shuffle-ablation
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=logs/shuffle_abl_%j.out
#SBATCH --error=logs/shuffle_abl_%j.err

# Train 3 CRAN-PM variants from scratch on Europe 2018, one per shuffle
# strategy. Each run is 3 epochs only — enough to compare convergence
# speed and early-epoch validation loss (the ablation goal).
#
# Outputs land in:
#   /scratch/.../topoflow_europe/checkpoints_shuffle_{wind,random,raster}/
#   /scratch/.../topoflow_europe/logs/multiscale_topoflow/version_*/
#
# Then the matching paper figure script reads TB logs to draw the
# 3-curve comparison.

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export MIOPEN_USER_DB_PATH=/tmp/miopen_shuffle_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES mkdir -p ${MIOPEN_USER_DB_PATH} 2>/dev/null || true
export PYTHONUNBUFFERED=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN

cd /scratch/project_462001140/ammar/eccv/topoflow_europe

run_one() {
  local mode="$1"
  echo ""
  echo "===================================================="
  echo "  Training shuffle_mode=${mode}  $(date)"
  echo "===================================================="
  export CKPT_DIR=/scratch/project_462001140/ammar/eccv/topoflow_europe/checkpoints_shuffle_${mode}
  mkdir -p "$CKPT_DIR"
  srun python3 -u scripts/train.py --config configs/ablation_shuffle_${mode}.yaml
}

run_one wind
run_one random
run_one raster

echo ""
echo "All three shuffle modes done: $(date)"
