# CRAN-PM

**Cross-Resolution Attention Network for high-resolution PM2.5 forecasting**

CRAN-PM is a multi-scale deep learning model for short-range (1-4 day)
PM2.5 surface concentration forecasting at high spatial resolution over
Europe. It fuses ERA5 meteorology, CAMS chemistry and high-resolution
elevation through a cross-resolution attention bridge.

This repository provides the reference implementation as a properly
packaged scientific software tool, intended for community use and as a
companion to the upcoming GMD model description paper.

## Installation

```bash
pip install cranpm
```

For GPU inference and full functionality:

```bash
pip install "cranpm[all]"
```

For development:

```bash
git clone https://github.com/AmmarKheder/cran-pm
cd cran-pm
pip install -e ".[dev,docs]"
pre-commit install
```

## Quick start

### Python API

```python
from cranpm import CRANPMForecaster

model = CRANPMForecaster.from_pretrained("AmmarKheder/cran-pm-europe-v3")

forecast = model.predict(
    era5=era5_dataset,
    cams=cams_dataset,
    elevation=elevation_dataset,
    lead_time=1,  # days ahead
)
```

### Command line

```bash
cranpm forecast \
    --date 2025-01-15 \
    --region europe \
    --lead-time 1 \
    --output forecast.nc
```

## GPU support

CRAN-PM ships with a dedicated GPU module supporting both NVIDIA CUDA
and AMD ROCm. See [docs/gpu.md](docs/gpu.md) for details.

| Backend | Status     | Tested on            |
|---------|------------|----------------------|
| CUDA    | supported  | A100, H100           |
| ROCm    | supported  | MI250X (LUMI)        |

## Documentation

Full documentation: https://ammarkheder.github.io/cran-pm/

## Citation

If you use CRAN-PM in your research, please cite the GMD paper (in
preparation) and the software via Zenodo. See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
