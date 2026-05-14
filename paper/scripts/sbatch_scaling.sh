#!/bin/bash -l
#SBATCH --job-name=cranpm-scaling
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=logs/scaling_%j.out
#SBATCH --error=logs/scaling_%j.err

# CRAN-PM GMD paper — strong + weak scaling on LUMI.
# Submit four times with --nodes={1,2,4,8} (corresponding to 8/16/32/64 GPUs)
# to fill in the scaling curve. We log per-step time to a JSON file the plot
# script then ingests.

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-db

# Create the MIOpen cache dir on every node (LUMI /tmp is node-local).
srun --ntasks-per-node=1 mkdir -p "$MIOPEN_USER_DB_PATH"

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

CONFIG=/scratch/project_462001140/ammar/eccv/cran-pm/configs/europe_multiscale.yaml
OUT_DIR=/scratch/project_462001140/ammar/eccv/cran-pm/paper/figures/scaling_${SLURM_NNODES}n
mkdir -p "$OUT_DIR"

# Run a fixed number of optimisation steps to compare wall-clock per step.
# We do NOT save checkpoints — this is a pure performance probe.
srun python -u -m cranpm.training.bench_train \
    --config "$CONFIG" \
    --max-steps 100 \
    --warmup-steps 20 \
    --log-json "$OUT_DIR/scaling_${SLURM_NNODES}n.json"

echo "Wrote $OUT_DIR/scaling_${SLURM_NNODES}n.json"
