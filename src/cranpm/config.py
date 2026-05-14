"""Centralised configuration: paths and runtime defaults via env vars.

Environment variables
---------------------
CRANPM_DATA_ROOT
    Root directory containing the zarr datasets (era5, cams, ghap, elevation).
    Default: ``$HOME/.cranpm/data``.

CRANPM_CACHE_DIR
    Directory for downloaded model checkpoints and intermediate caches.
    Default: ``$HOME/.cranpm/cache``.

CRANPM_HF_TOKEN
    HuggingFace token used for downloading private model checkpoints.
    Falls back to ``HF_TOKEN`` if unset.

CRANPM_DEVICE
    Default device for inference (``cuda``, ``cuda:0``, ``cpu``).
    Auto-detected when unset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Paths:
    """File-system layout for CRAN-PM artefacts."""

    data_root: Path
    cache_dir: Path

    @property
    def era5_dir(self) -> Path:
        return self.data_root / "era5_europe_daily"

    @property
    def cams_dir(self) -> Path:
        return self.data_root / "cams_europe"

    @property
    def ghap_dir(self) -> Path:
        return self.data_root / "ghap_pm25_europe_daily"

    @property
    def elev_coarse_path(self) -> Path:
        return self.data_root / "elevation" / "elevation.zarr"

    @property
    def elev_hires_path(self) -> Path:
        return self.data_root / "elevation" / "gmted2010_europe.zarr"

    @property
    def eea_zarr_path(self) -> Path:
        return self.data_root / "eea_stations" / "eea_pm25_daily.zarr"

    @property
    def checkpoints_dir(self) -> Path:
        return self.cache_dir / "checkpoints"


def get_paths() -> Paths:
    """Return resolved data / cache paths. Created on demand."""
    home = Path.home()
    data_root = _env_path("CRANPM_DATA_ROOT", home / ".cranpm" / "data")
    cache_dir = _env_path("CRANPM_CACHE_DIR", home / ".cranpm" / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Paths(data_root=data_root, cache_dir=cache_dir)


def get_hf_token() -> str | None:
    return os.environ.get("CRANPM_HF_TOKEN") or os.environ.get("HF_TOKEN")


def get_default_device() -> str:
    explicit = os.environ.get("CRANPM_DEVICE")
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
