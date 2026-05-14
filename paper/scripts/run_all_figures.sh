#!/usr/bin/env bash
# Regenerate every figure that can be produced from existing data
# (i.e., does not require new compute on LUMI).
#
# Run after pulling new evaluation outputs into evaluation_2022/.

set -euo pipefail

PYTHON=/scratch/project_462000640/ammar/aq_net2/venv_pytorch_rocm/bin/python
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPTS_DIR"

scripts=(
    fig_robinson_study_domain.py
    fig_software_stack.py
    fig_eea_stations_europe.py
    fig_annual_mean_europe.py
    fig_best_day_comparison.py
    fig_method_comparison.py
    fig_lead_time_curves.py
    fig_country_heatmap.py
    fig_ablation_bars.py
)

for s in "${scripts[@]}"; do
    echo "==> $s"
    "$PYTHON" "$s"
done

echo "All figures regenerated -> /scratch/project_462001140/ammar/eccv/cran-pm/paper/figures/"
