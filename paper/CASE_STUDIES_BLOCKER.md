# Case studies — pending data preparation

The three case studies (Saharan dust March 2022, Iberian fires July 2022,
Polish/Slovak winter heating January 2022) cannot be sbatch'd in their
current form because the Europe-cropped daily zarrs referenced by the
inference pipeline do **not** exist on the current file system:

```
MISSING:
  /scratch/project_462001140/ammar/eccv/data/zarr/era5_europe_daily/<year>.zarr
  /scratch/project_462001140/ammar/eccv/data/zarr/ghap_pm25_europe_daily/<year>.zarr
  /scratch/project_462001140/ammar/eccv/data/zarr/cams_europe/<year>.zarr

PRESENT:
  /scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily/<year>.zarr
  /scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily/<year>.zarr
  /scratch/project_462001140/ammar/eccv/data/zarr/cams_global_daily/<year>.zarr
  /scratch/project_462001140/ammar/eccv/data/zarr/cams_analysis_04_daily/<year>.zarr
```

Two ways to unblock the case-study pipeline:

## Option 1 — re-run the existing prepare scripts (recommended)

The original ECCV pipeline that produced `evaluation_2022/` ran on
Europe-cropped zarrs. The scripts that built them are in
`/scratch/project_462001140/ammar/eccv/topoflow_europe/scripts/` and
include `download_*.py` for raw data plus a tile/crop step.
Re-running them on 2022 alone for the three case windows would take
roughly 4-6 CPU-hours (no GPU needed).

```bash
cd /scratch/project_462001140/ammar/eccv/topoflow_europe/scripts
python build_era5_europe_zarr.py --year 2022   # ~1 GPU-h
python build_ghap_europe_zarr.py --year 2022
python build_cams_europe_zarr.py --year 2022
```

(Adjust the actual script names — they may be named differently in your
checkout.)

## Option 2 — point the case-study runner at the global zarrs

A new `paper/scripts/run_case_study.py` could open the global zarrs,
crop to the European domain on the fly, and feed the patches to
`CRANPMForecaster.predict`. This would avoid duplicating ~50 GB of data
but adds a non-trivial xarray indexing step. To do.

Until either option lands, the case-study sections of the paper remain
narrative-only with `\\includegraphics` placeholders. The physics
ablation table (Section 8b) and the GPU benchmarks (Section 5) do not
depend on this and run independently.
