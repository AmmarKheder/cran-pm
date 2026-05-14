# Figure inventory and status

Auto-managed: regenerate by running the `scripts/fig_*.py` scripts.

## Generated and committed (DONE)

| Slug                            | Source data                                                       | Section  |
|---------------------------------|-------------------------------------------------------------------|----------|
| `fig_robinson_study_domain`     | Master_Validation_2025.npz (12 k stations, Robinson world view)   | §1, §2   |
| `fig_software_stack`            | matplotlib block diagram, no data                                 | §4       |
| `fig_eea_stations_europe`       | Master_Validation_2025.npz, Europe LAEA, 2 967 stations           | §6       |
| `fig_annual_mean_europe`        | annual_mean_t1.npz (4192×6992, downsampled 4×, 3-panel LAEA)      | §6       |
| `fig_best_day_comparison`       | best_day_maps.npz (2022-05-31, 3-panel LAEA)                      | §6       |
| `fig_method_comparison`         | main_table_*.json (7 methods × 3 lead times, RMSE + R² bars)      | §6       |
| `fig_lead_time_curves`          | main_table_*.json (RMSE + R² vs T+1..T+3, line plot)              | §6       |
| `fig_country_heatmap`           | country_mae_cache.npz (4 methods × 9 EU countries)                | §6       |
| `fig_ablation_bars`             | ablation_table_full.json (5 cross-attention variants)             | §8       |

## To produce — physics ablation battery (new addition)

| Slug                                  | What to run                                                        | Est. GPU-days |
|---------------------------------------|--------------------------------------------------------------------|---------------|
| `fig_physics_ablations_bars`          | bars of RMSE Δ per intervention for CRAN-PM                        | 0.5           |
| `fig_physics_ablations_heatmap`       | model × intervention heatmap (CRAN-PM, TopoFlow, ClimaX, …)        | 2.0           |
| `table_physics_ablations.tex`         | LaTeX big table T-Phys-1 (auto-rendered)                           | included      |

Submission: `sbatch sbatch_physics_ablations.sh` (single GPU, ~8h for the full multi-model run on 30 days).

## To produce — needs new compute on LUMI

| Slug                            | What to run                                              | Est. GPU-days | Status      |
|---------------------------------|----------------------------------------------------------|---------------|-------------|
| `fig_gpu_benchmarks`            | DONE — see `bench_rocm.json` (MI250X, 0.72 maps/s bf16)  | 0             | ✅ generated |
| `fig_gpu_benchmarks_cuda`       | run on a CUDA box separately, JSON merged                | 0.5           | doc only    |
| `fig_scaling`                   | `sbatch sbatch_scaling.sh` × 4 (1/2/4/8 nodes)           | 8             | sbatch ready|
| `fig_case_saharan_dust`         | `sbatch sbatch_case_study.sh saharan-dust 2022-03-14 2022-03-19` | 2     | sbatch ready|
| `fig_case_iberian_fires`        | `sbatch sbatch_case_study.sh iberian-fires 2022-07-14 2022-07-22` | 3    | sbatch ready|
| `fig_case_polish_winter`        | `sbatch sbatch_case_study.sh polish-winter 2022-01-08 2022-01-18` | 2    | sbatch ready|
| `fig_uncertainty_calibration`   | new script using MC dropout × 10 forward passes / day    | 4             | TODO        |

## To produce — pure scripting from existing data (no new compute)

| Slug                            | What needs writing                                       | Est. effort |
|---------------------------------|----------------------------------------------------------|-------------|
| `fig_per_station_rmse_europe`   | recompute per-station RMSE from predictions_t1.zarr      | 1 hour      |
| `fig_scatter_density`           | density scatter from predictions_t1.zarr (~100M pixels)  | 1 hour      |
| `fig_input_data_flow`           | 4-panel ERA5/CAMS/GHAP/elevation snapshot for fixed day  | 1 hour      |
| `fig_seasonal_rmse_heatmap`     | aggregate predictions_t1.zarr by month                   | 1 hour      |

## Reused from ECCV (already in `figures_eccv_source/`)

- `TEASER_ECCV.pdf` -> teaser, possibly trimmed for GMD layout
- `fig_architecture_overview.pdf` -> §3 architecture overview
- `network.jpg` -> alt schematic
- `fig3_spatial_final.pdf`, `fig4_temporal_curves_v3.png` -> consider re-style for GMD

## Layout sanity checks before submission

- All PDF pages render in `xelatex` and `pdflatex`.
- Each figure has a meaningful caption (one sentence overview + one sentence reading guide).
- Color scheme consistent (METHOD_COLORS in `cranpm_plots.py`).
- All Europe maps in LAEA (EPSG:3035) projection.
- Robinson reserved for the global domain figure.
- No raster figure smaller than 200 dpi at print size.
