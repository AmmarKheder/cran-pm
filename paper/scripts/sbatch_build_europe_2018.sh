#!/bin/bash -l
#SBATCH --job-name=build-europe-2018
#SBATCH --account=project_462001140
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_europe_%j.out
#SBATCH --error=logs/build_europe_%j.err

set -euo pipefail
mkdir -p logs

module load LUMI/25.03
source /scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/activate

cd /scratch/project_462001140/ammar/eccv/topoflow_europe

# 2018 + 2021 covers the minimum (1 train year + 1 val year) for short ablation training.
srun python scripts/build_europe_zarrs_from_global.py --years 2018 2021

echo "Done."
ls -lah /scratch/project_462001140/ammar/eccv/data/zarr/era5_europe_daily 2>/dev/null | head -5
ls -lah /scratch/project_462001140/ammar/eccv/data/zarr/ghap_pm25_europe_daily 2>/dev/null | head -5
ls -lah /scratch/project_462001140/ammar/eccv/data/zarr/cams_europe 2>/dev/null | head -5
