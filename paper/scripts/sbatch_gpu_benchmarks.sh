#!/bin/bash -l
#SBATCH --job-name=cranpm-gpu-bench
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gpu_bench_%j.out
#SBATCH --error=logs/gpu_bench_%j.err

# CRAN-PM GMD paper — single-GPU inference benchmarks (ROCm side).
# CUDA-side numbers are produced separately on a CUDA-equipped machine
# (e.g., a workstation or HuggingFace inference endpoint) using:
#   python -m cranpm.gpu.bench --backend cuda --output bench_cuda.json
# and merged into the same plot.

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-db
mkdir -p "$MIOPEN_USER_DB_PATH"

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

OUT=/scratch/project_462001140/ammar/eccv/cran-pm/paper/figures/bench_rocm.json
CHECKPOINT=/pfs/lustrep1/scratch/project_462001140/ammar/eccv/topoflow_europe/checkpoints_v3_best524/topoflow-016.ckpt

srun python -m cranpm.gpu.bench \
    --backend rocm \
    --checkpoint "$CHECKPOINT" \
    --precision bf16 fp16 fp32 \
    --batch-sizes 1 2 4 8 \
    --warmup-iter 5 \
    --measure-iter 50 \
    --output "$OUT"

echo "Wrote $OUT"
