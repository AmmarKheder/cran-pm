#!/bin/bash -l
#SBATCH --job-name=cranpm-physics-ablations
#SBATCH --account=project_462001140
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/physics_%j.out
#SBATCH --error=logs/physics_%j.err

# CRAN-PM GMD paper — physics-ablation battery.
# Runs 13 ablations × N samples on the v3 checkpoint, then renders the
# paper figures and LaTeX table.

set -euo pipefail
mkdir -p logs

module load LUMI/25.03 partition/G rocm/6.0.3
export MIOPEN_USER_DB_PATH=/tmp/$USER/miopen-db
mkdir -p "$MIOPEN_USER_DB_PATH"

source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

CHECKPOINTS_DIR=/pfs/lustrep1/scratch/project_462001140/ammar/eccv/topoflow_europe/paper_checkpoints
PREDICTIONS=/pfs/lustrep1/scratch/project_462001140/ammar/eccv/topoflow_europe/evaluation_2022/predictions_t1.zarr
CAMS_FORECAST=/scratch/project_462001140/ammar/eccv/data/zarr/cams_forecast_europe_2022.zarr
OUTDIR=/scratch/project_462001140/ammar/eccv/cran-pm/paper/figures

cd /scratch/project_462001140/ammar/eccv/cran-pm

# Run the full ablation battery on every ML baseline + CAMS reference.
# CAMS itself is not perturbed (would require re-running the operational
# CTM at $10^4$ CPU-hours per intervention); we only report its baseline
# skill on the same days for context.
srun python paper/scripts/physics_ablations.py \
    --models cranpm topoflow climax earthformer simvp convlstm \
    --checkpoints-dir "$CHECKPOINTS_DIR" \
    --predictions "$PREDICTIONS" \
    --cams-forecast "$CAMS_FORECAST" \
    --output-dir "$OUTDIR" \
    --n-samples 30 \
    --lead-time 1 \
    --precision bf16

srun python paper/scripts/fig_physics_ablations_table.py \
    --results "$OUTDIR/physics_ablations_all.json" \
    --output-dir "$OUTDIR"

echo "Physics ablations done."
