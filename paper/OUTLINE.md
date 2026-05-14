# CRAN-PM v1.0 — GMD model description paper outline

**Target journal**: Geoscientific Model Development (GMD), Copernicus Publications.
**Paper type**: Model description paper (`Geosci. Model Dev.`).
**Companion to**: Kheder et al. 2026 (ECCV, in preparation).
**Tool DOI**: Zenodo (TBD, pinned to cran-pm v0.1.0 release).
**Suggested title**: *"CRAN-PM v1.0: a multi-scale Vision Transformer for high-resolution PM2.5 forecasting over Europe"*.

Estimated length: 25-35 pages (GMD has no strict limit; model description papers
typically run 20-40 pages including code/data section).

Color legend below:
- **REUSE** = direct port from ECCV with light edits
- **REWRITE** = same scientific content, geosciences-oriented framing
- **NEW** = entirely new content for GMD

---

## 1. Introduction (REWRITE — 1.5 pages)

ECCV opens with "Vision Transformers have transformed image processing..."  — that's
a ML-conference framing. GMD wants:
- Geosciences problem framing first: PM2.5 health burden in Europe (EEA reports, ~250k premature deaths/year), regulatory needs (EU AAQD revision 2024).
- State of operational forecasting: CAMS (40 km), national systems (e.g., LOTOS-EUROS, EMEP), their resolution limits.
- Motivation for ML: bridging physical models and ground-station observations at 1 km.
- Why a new tool: open-source, reproducible, GPU-accessible alternative.
- Contributions framed as a software contribution: model + reproducible pipeline + evaluation harness + open data tooling + GPU benchmarks.
- One-paragraph paper roadmap.

---

## 2. Background and related software (REWRITE — 2 pages)

Replace the ECCV "Related Work" (which compares to ML papers like Swin/ViT) with
a software/community survey:
- Operational atmospheric chemistry models (CAMS, LOTOS-EUROS, EMEP, WRF-Chem) — open vs proprietary, resolutions, computational cost.
- ML-based AQ forecasting tools: ClimaX, GraphCast, Pangu, Aurora — what's open, what's not, license, resolution.
- High-resolution PM2.5 retrievals: GHAP (1 km, retrospective), CAMS (40 km).
- Position CRAN-PM in this landscape (table comparing open-source status, resolution, lead time, GPU support).

**NEW**: Table comparing software ecosystem (license, language, GPU backend, training/inference reproducibility, data accessibility) — most distinctive section vs ECCV.

---

## 3. Model description (REUSE + light edits — 6 pages)

This is the canonical "Section 2" of a GMD model description paper. Reuse most of
ECCV's "Method" section, with restructured headings:

### 3.1 Forecasting problem and conventions
Inputs (ERA5 35 ch + CAMS 6 polluants + GHAP 1 km PM2.5 + GMTED2010 elevation),
output (PM2.5 at 1 km, lead times 1-4 days), variable definitions, units, time
zones, calendar conventions.

### 3.2 Architecture overview (REUSE Fig 1)
Dual-branch design, cross-resolution attention bridge, decoder, delta-prediction.

### 3.3 Global meteorological branch (REUSE)
ViT, patch embed via F.unfold + Linear (mention the MIOpen workaround in §6).

### 3.4 Local fine-resolution branch (REUSE)
TopoFlow blocks with elevation bias, wind scanning.

### 3.5 Cross-resolution attention bridge (REUSE)
Equations from ECCV §2, wind-guided attention bias.

### 3.6 Delta prediction and zero-init decoder (REUSE)
Persistence baseline at init, CNN decoder progressive upsampling.

### 3.7 Training objective (REUSE)
Land-only MSE, FFL spectral loss, station loss, ablation justifying each term.

---

## 4. Implementation and software design (NEW — 4 pages)

The heart of the GMD paper. Not present in ECCV.

### 4.1 Package layout and public API
- `src/cranpm/` layout, `pyproject.toml`, semantic versioning.
- Public API: `CRANPMForecaster.from_pretrained(...)`, `.predict(...)`.
- CLI: `cranpm forecast`, `cranpm train`, `cranpm benchmark`, `cranpm download`.
- Configuration via env vars (`CRANPM_DATA_ROOT`) and YAML (`europe_multiscale.yaml`).

### 4.2 Data pipeline
- Zarr layout (chunking strategy, normalization stats), patch extractor.
- Download scripts for ERA5/CAMS/GHAP/GMTED2010 via Copernicus CDS.
- Caching, resume logic.

### 4.3 GPU backend support
- Single-GPU inference path (bf16 / fp16, torch.compile).
- Distributed training via PyTorch Lightning DDP (portable beyond LUMI).
- CUDA + ROCm dual support, Dockerfile per backend.
- ROCm-specific issues encountered and worked around (MIOpen Conv2d numerical
  bug → F.unfold + Linear; MIOpen cache on shared FS).

### 4.4 Reproducibility infrastructure
- pytest suite (unit + integration), GitHub Actions CI.
- Model checkpoint format: HuggingFace safetensors, model card auto-generated.
- Seeded training, deterministic eval mode.
- Pinned dependency set (`requirements.lock`).

### 4.5 Deployment surfaces
- HuggingFace Hub model repo (pretrained weights).
- HuggingFace Space (Gradio demo for non-coders).
- Zenodo release for citable DOI.

---

## 5. Computational performance (NEW — 2 pages)

Critical for GMD readers running their own science. Not in ECCV.

### 5.1 Inference benchmarks
Table: throughput (maps/sec), peak memory, latency vs lead-time vs precision (fp32 / bf16 / fp16),
on:
- 1× A100 80 GB (CUDA reference)
- 1× MI250X GCD (ROCm, LUMI)
- 1× consumer GPU (RTX 4090) — for accessibility
- CPU fallback (timing only, no scaling claim)

### 5.2 Training scaling
Strong + weak scaling on LUMI: 1 → 8 → 32 → 64 GPUs. Time-per-epoch, GPU utilisation,
NCCL bandwidth.

### 5.3 Energy / cost
Estimate kWh per training run (LUMI is ~100% renewable — interesting hook), per-forecast
energy. Compare to CAMS operational cost (rough order of magnitude).

---

## 6. Evaluation (REWRITE — 5 pages)

Reuse ECCV results but extend significantly:

### 6.1 Evaluation protocol
- Datasets: EEA stations (training years 2017-2020, val 2021, test 2022).
- Metrics: RMSE, MAE, R², bias, correlation, hit-rate for AQI thresholds.
- New: stratification by season, country, station-type (urban/rural/traffic),
  altitude bin (low/mid/high).

### 6.2 Comparison to baselines (REUSE ECCV table)
ConvLSTM, ClimaX, Earthformer, SimVP, persistence, climatology, CAMS reanalysis.

### 6.3 Comparison to operational forecasts (NEW — pending compute)
**Important new addition**: compare CRAN-PM forecasts at T+1...T+4 against the
**operational CAMS regional forecast** (not just reanalysis). This is what
operational users care about. Requires downloading CAMS forecast archive 2022.

### 6.4 Multi-year robustness (NEW — pending compute)
Re-run inference on test years 2020, 2021, 2023 (in addition to 2022). Annual
RMSE table. Show seasonal stability.

### 6.5 Uncertainty estimation (NEW — pending compute)
Either MC dropout (cheap) or 5-member ensemble via different seeds (expensive).
Calibration plot.

---

## 7. Case studies (NEW — 4 pages, ~1.5 page each)

These are the soul of the GMD paper. Pick representative episodes from 2022.

### 7.1 Saharan dust intrusion (14-19 March 2022)
- Episode synopsis: Calima reaching France/UK, satellite imagery from MODIS, WHO threshold breaches.
- CRAN-PM forecast vs CAMS vs persistence vs ground truth (EEA).
- Show value of global branch capturing long-range advection.
- Spatial maps + station time series for ~6 selected stations along trajectory.

### 7.2 Iberian Peninsula wildfires (summer 2022)
- Worst wildfire season on record in Spain/Portugal.
- Sub-period: 14-22 July 2022 (Sierra de la Culebra fire).
- Show the local branch's ability to capture sharp PM2.5 plumes near burn area.
- Compare to CAMS (which under-resolves emissions).

### 7.3 Polish/Slovak winter heating episode (January 2022)
- Cold snap drives residential coal/biomass burning.
- Stations in Krakow, Katowice, Ostrava.
- Model captures urban-scale gradients invisible to coarse CAMS.
- Tie to EU air quality directive monitoring obligations.

Common figure layout per case study: 2x2 panel = (true PM2.5, CRAN-PM, CAMS, persistence)
+ time series at 4-6 ground stations.

---

## 8b. Physics ablation (NEW for GMD — 3 pages)

The strongest single addition of the GMD paper vs the ECCV submission.
Test-time interventions on each physical prior, applied to all ML
baselines + CAMS as physics reference. Three big tables:

**T-Phys-1 (per-CRAN-PM ablation)**: 13 interventions × {RMSE, MAE, R², bias, Δ vs baseline}.

**T-Phys-2 (multi-model heatmap)**: 6 ML models × 12 interventions, colour = RMSE Δ. Reveals which models are physics-aware.

**T-Phys-3 (stratified)**: per-intervention RMSE Δ on {mountain stations, coastal, high-wind days, calm days, plume episodes}.

WRF-Chem comparison: discussed in §2 with explicit caveat (no operational European WRF-Chem run at 1 km available for 2022; re-running it would cost ~$10^5$ CPU-hours).

Generated by: `paper/scripts/sbatch_physics_ablations.sh` → `paper/scripts/physics_ablations.py` → `paper/scripts/fig_physics_ablations_table.py`.

## 8. Ablation study (REUSE ECCV ablation — 1.5 page)

Keep ECCV ablation table. Drop the most ML-specific ablations (variant FiLM /
concat / etc.). Keep:
- No cross-attention vs cross-attention
- No elevation bias vs with
- No wind bias vs with
- No delta prediction vs with

---

## 9. Limitations (REWRITE — 1 page)

Reuse ECCV's limitations list, expand for tool perspective:
- Domain limited to Europe (model not trained outside).
- PM2.5 only — not yet PM10, NO2, O3 (future work).
- 1-4 day lead time (no longer-range forecasting).
- Requires CAMS + ERA5 inputs (CDS account needed for end users).
- Relies on GHAP for the t-1 PM2.5 input — limited operational availability.
- GPU recommended but CPU possible (slow).
- Calibration not yet ensemble-based.

---

## 10. Code and data availability (NEW — required by GMD — 0.5 page)

GMD-mandatory section. Lists every artifact with DOI/URL.
- **Source code**: GitHub repo + Zenodo DOI for v0.1.0 release.
- **Model weights**: HuggingFace Hub repo `AmmarKheder/cran-pm-europe-v3`, pinned.
- **Training data**: ERA5 (CDS), CAMS (CDS), GHAP (Wei et al. 2023, Zenodo), GMTED2010 (USGS), EEA (EEA data portal).
- **Evaluation outputs**: zarr archives of CRAN-PM forecasts on test years, deposited
  at Zenodo for inspection.
- **Docker images**: `cranpm/cuda:0.1.0`, `cranpm/rocm:0.1.0` on Docker Hub.

---

## 11. Conclusions (REWRITE — 0.5 page)

Software contribution framing: CRAN-PM v1.0 is open, reproducible, GPU-accessible,
extensible. Encourages community use for evaluation, fine-tuning, regional adaptation.
Brief future work tease (PM10, NO2, ensemble, regional fine-tuning).

---

## Acknowledgements + funding

- LUMI compute (CSC Finland, EuroHPC project_462001140).
- Funding sources: LUT, AMC-Lahti, etc.
- AMD support (Samuel Antão).

---

## Required new experiments for GMD (compute plan)

| ID    | Description                                          | Est. GPU-days | Owner | Priority |
|-------|------------------------------------------------------|---------------|-------|----------|
| EXP-1 | GPU benchmark suite (1 GPU, 4 backends)              | 2             | Ammar | P0       |
| EXP-2 | Multi-year inference (2020, 2021, 2023)              | 6             | Ammar | P0       |
| EXP-3 | Download + evaluation against CAMS forecast 2022     | 4             | Ammar | P1       |
| EXP-4 | Saharan dust case study (data + figures)             | 2             | Ammar | P0       |
| EXP-5 | Iberian wildfires case study (data + figures)        | 3             | Ammar | P0       |
| EXP-6 | Polish/Slovak winter case study (data + figures)     | 2             | Ammar | P0       |
| EXP-7 | MC dropout uncertainty (10 forward passes / day)     | 4             | Ammar | P2       |
| EXP-8 | Strong + weak scaling on 1/8/32/64 GPUs              | 8             | Ammar | P1       |
| EXP-9 | Stratified evaluation by season/country/altitude     | 1 (CPU mostly)| Ammar | P0       |
| EXP-10| Re-train v3 with seed 42 for repro check             | 12            | Ammar | P3       |

Total P0+P1: ~28 GPU-days. P0+P1+P2: ~32. Comfortable within 100-day budget.

---

## Figures (~20 figures total, mix REUSE / NEW now / NEW pending compute)

Projection conventions: **Robinson** for global context, **LAEA (EPSG:3035)** for
all Europe regional maps, **PlateCarree** only for raw-grid display.

| #     | Figure                                                       | Projection     | Source / status                                          |
|-------|--------------------------------------------------------------|----------------|----------------------------------------------------------|
| F1    | Teaser (Europe 1 km forecast 2022-01-25, EEA overlays)       | LAEA           | REUSE ECCV                                               |
| F2    | Architecture overview (dual-branch + cross-attention)        | -              | REUSE ECCV                                               |
| F3    | Software stack diagram (data → loader → model → output)      | -              | NEW (matplotlib block diagram, no data)                  |
| F4    | **Robinson world map: study domain + OOD test regions**      | Robinson       | NEW (DONE — uses Master_Validation_2025 station coords)  |
| F5    | EEA station coverage Europe (urban/rural/traffic)            | LAEA           | NEW (pending station type metadata)                      |
| F6    | Input data flow: 4 panels (ERA5 wind, CAMS, GHAP, elevation) | LAEA           | NEW (pending)                                            |
| F7    | GPU benchmark bars (throughput / memory / latency × 4 GPUs)  | -              | NEW (EXP-1, ~2 GPU-days)                                 |
| F8    | Strong + weak scaling on LUMI (1, 8, 32, 64 GPUs)            | -              | NEW (EXP-8, ~8 GPU-days)                                 |
| F9    | **Annual mean PM2.5 — observed vs CRAN-PM vs bias**          | LAEA × 3       | NEW (DONE — annual_mean_t1.npz)                          |
| F10   | Per-country MAE heatmap (4 methods × 8 countries)            | -              | NEW (DONE — country_mae_cache.npz)                       |
| F11   | Country / season stratified RMSE heatmap (extended)          | -              | NEW (EXP-9, light)                                       |
| F12   | Method comparison bars (RMSE + R² × 3 lead times)            | -              | NEW (DONE — main_table_*.json)                           |
| F13   | Lead-time degradation curves (RMSE + R² vs T+1..T+3)         | -              | NEW (DONE — main_table_*.json)                           |
| F14   | Best forecast day 2022 — observed vs CRAN-PM vs bias         | LAEA × 3       | NEW (DONE — best_day_maps.npz)                           |
| F15   | Temporal curves at 6 selected stations (multi-method)        | -              | REUSE ECCV (regenerate with all 8 methods)               |
| F16   | Scatter density obs vs predicted (annual + per-season)       | -              | NEW (regenerate from predictions_t1.zarr)                |
| F17   | **Saharan dust intrusion (March 2022) case study**           | LAEA × 4 + ts  | NEW (EXP-4, ~2 GPU-days)                                 |
| F18   | **Iberian wildfires (July 2022) case study**                 | LAEA × 4 + ts  | NEW (EXP-5, ~3 GPU-days)                                 |
| F19   | **Polish/Slovak winter heating (January 2022) case study**   | LAEA × 4 + ts  | NEW (EXP-6, ~2 GPU-days)                                 |
| F20   | Ablation spatial maps (full vs no-cross-attn vs no-elev)     | LAEA × 3       | REUSE ECCV                                               |
| F21   | Ablation bar chart (RMSE delta per component)                | -              | REUSE ECCV (re-style)                                    |
| F22   | MC-dropout uncertainty calibration plot                      | -              | NEW (EXP-7, ~4 GPU-days)                                 |
| F23   | OOD generalization Robinson map (USA/Canada/India scatter)   | Robinson       | NEW (regenerate from existing OOD outputs)               |
| F24   | Per-station RMSE Europe map (color = RMSE, size = n_obs)     | LAEA           | NEW (pending per-station eval rerun, ~1 day)             |

---

## Tables (~7 tables)

| #   | Table                                              | Source        |
|-----|----------------------------------------------------|---------------|
| T1  | Software ecosystem comparison (license/lang/GPU)   | NEW           |
| T2  | Input variable list (ERA5/CAMS/GHAP details)       | NEW           |
| T3  | Inference benchmark by hardware                    | NEW (EXP-1)   |
| T4  | Main results vs baselines (RMSE/MAE/R² per LT)     | REUSE ECCV    |
| T5  | Multi-year RMSE                                    | NEW (EXP-2)   |
| T6  | CRAN-PM vs CAMS forecast operational               | NEW (EXP-3)   |
| T7  | Stratified RMSE (season/country/altitude)          | NEW (EXP-9)   |
| T8  | Ablation study                                     | REUSE ECCV    |

---

## Submission checklist (GMD-specific)

- [ ] Manuscript in Copernicus LaTeX template (`copernicus.cls`)
- [ ] Code repository with Zenodo DOI
- [ ] Model weights with persistent identifier (HF Hub or Zenodo)
- [ ] Data sources documented with DOI/URL each
- [ ] Reproducibility section: install, run, test commands
- [ ] License is OSI-approved (MIT ✓)
- [ ] All figures have alt text in caption
- [ ] No paywalled dependencies
- [ ] Author contributions statement (CRediT taxonomy)
- [ ] Conflict of interest statement
- [ ] Data availability statement explicit per dataset
