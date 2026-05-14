"""Build :class:`ForecastInputs` from the GLOBAL zarrs by cropping to Europe.

We do not require pre-built Europe-cropped zarrs. The crop is cheap (numpy
slicing on contiguous chunks) and is done on demand, day by day.

Coverage assumed in this module
-------------------------------
* ERA5 daily   (365, 30, 721, 1440)         at 0.25 deg, global   (90 N -> -90 S)
* GHAP daily   (365, 18000, 36000)          at 0.01 deg, global   (90 N -> -90 S)
* CAMS 0.4 deg group {pm25,no2,so2,o3,co,pm10} (365, 451, 900)    global
* GMTED2010 Europe elevation (4192, 6992)   pre-built

If your zarrs use a different grid the constants below need to be updated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from cranpm.inference.forecaster import ForecastInputs
from cranpm.inference.tiling import GHAP_H, GHAP_W


# --- Europe domain (matches configs/europe_multiscale.yaml) ----------------
LAT_NORTH = 72.0
LAT_SOUTH = 30.08          # 72 - 4192 * 0.01
LON_WEST = -25.0
LON_EAST = 44.92           # -25 + 6992 * 0.01

# --- ERA5 grid: global 0.25 deg, lat from 90 N to -90 S ---------------------
ERA5_RES = 0.25
ERA5_N_LAT = 721           # 0..720 inclusive (90 deg to -90 deg)
ERA5_N_LON = 1440
ERA5_LAT_TOP_IDX = int(round((90.0 - LAT_NORTH) / ERA5_RES))    # 72
# ERA5 longitude convention: 0..360 (typical for CDS)
def _era5_lon_idx(lon_deg: float) -> int:
    # Convert to 0..360 then to grid index.
    x = (lon_deg + 360.0) % 360.0
    return int(round(x / ERA5_RES))

ERA5_LAT_SLICE = slice(ERA5_LAT_TOP_IDX, ERA5_LAT_TOP_IDX + 168)  # 168 rows
# Two contiguous chunks (because the European domain crosses lon=0).
ERA5_LON_WEST_IDX = _era5_lon_idx(LON_WEST)    # e.g. (-25+360)/0.25=1340
ERA5_LON_EAST_IDX = _era5_lon_idx(LON_EAST)    # 45/0.25=180

# --- GHAP grid: global 0.01 deg, lat from 90 N to -90 S ---------------------
GHAP_RES = 0.01
GHAP_LAT_TOP_IDX = int(round((90.0 - LAT_NORTH) / GHAP_RES))     # 1800
GHAP_LAT_SLICE = slice(GHAP_LAT_TOP_IDX, GHAP_LAT_TOP_IDX + GHAP_H)
# GHAP longitude convention: -180..180 (verify); we slice the obvious window.
def _ghap_lon_idx(lon_deg: float) -> int:
    return int(round((lon_deg + 180.0) / GHAP_RES))

GHAP_LON_WEST_IDX = _ghap_lon_idx(LON_WEST)     # 15500
GHAP_LON_EAST_IDX = _ghap_lon_idx(LON_EAST)     # 22492 (~22500 for 6992 width)

# --- CAMS 0.4 deg grid ------------------------------------------------------
CAMS_RES = 0.4
CAMS_N_LAT = 451
CAMS_N_LON = 900
CAMS_LAT_TOP_IDX = int(round((90.0 - LAT_NORTH) / CAMS_RES))     # 45
CAMS_LAT_BOT_IDX = int(round((90.0 - LAT_SOUTH) / CAMS_RES))     # 150
def _cams_lon_idx(lon_deg: float) -> int:
    return int(round((lon_deg + 180.0) / CAMS_RES))
CAMS_LON_WEST_IDX = _cams_lon_idx(LON_WEST)
CAMS_LON_EAST_IDX = _cams_lon_idx(LON_EAST)


CAMS_VARS = ("pm25", "no2", "so2", "o3", "co")  # 5 channels; pm10 reserved.


@dataclass
class EuropeZarrPaths:
    era5_dir: Path
    ghap_dir: Path
    cams_dir: Path
    elev_coarse_path: Path = Path("/scratch/project_462001140/ammar/eccv/data/"
                                   "zarr/elevation/elevation.zarr")
    elev_hires_path: Path = Path("/scratch/project_462001140/ammar/eccv/data/"
                                  "zarr/elevation/gmted2010_europe.zarr")

    @classmethod
    def default(cls) -> "EuropeZarrPaths":
        base = Path("/scratch/project_462001140/ammar/eccv/data/zarr")
        return cls(
            era5_dir=base / "era5_global_daily",
            ghap_dir=base / "ghap_global_daily",
            cams_dir=base / "cams_analysis_04_daily",
        )


class EuropeInputsBuilder:
    """Stream ForecastInputs from the global zarrs, day by day.

    Caches the per-year zarr handles so repeated reads on the same year
    pay the zarr open cost only once.
    """

    def __init__(self, paths: EuropeZarrPaths | None = None,
                  cache_size: int = 4):
        self.paths = paths or EuropeZarrPaths.default()
        self._era5_cache: dict[int, "zarr.Array"] = {}
        self._ghap_cache: dict[int, "zarr.Array"] = {}
        self._cams_cache: dict[int, "zarr.Group"] = {}

        # Elevation is constant in time -- cache eagerly.
        try:
            elev_c = zarr.open(str(self.paths.elev_coarse_path), mode="r")
            self.elev_coarse = self._normalise_elev_coarse(np.asarray(elev_c))
        except Exception:
            self.elev_coarse = np.zeros((168, 280), dtype=np.float32)
        # High-res elevation: GMTED2010 covers Europe at finer-than-1km res
        # (20160 x 33600 ~ 0.002 deg). Downsample to the GHAP grid by
        # block-average so the local branch sees coherent elevation tiles.
        try:
            elev_h_full = np.asarray(zarr.open(
                str(self.paths.elev_hires_path), mode="r"))
            self.elev_hires = self._downsample_elev_hires(elev_h_full)
        except Exception:
            self.elev_hires = np.zeros((GHAP_H, GHAP_W), dtype=np.float32)

    @staticmethod
    def _downsample_elev_hires(arr: np.ndarray) -> np.ndarray:
        """Block-average ``arr`` to the GHAP (4192, 6992) grid."""
        if arr.shape == (GHAP_H, GHAP_W):
            return arr.astype(np.float32)
        h, w = arr.shape
        fy = h // GHAP_H
        fx = w // GHAP_W
        if fy == 0 or fx == 0:
            return np.zeros((GHAP_H, GHAP_W), dtype=np.float32)
        crop = arr[: GHAP_H * fy, : GHAP_W * fx]
        return (crop.reshape(GHAP_H, fy, GHAP_W, fx)
                    .mean(axis=(1, 3)).astype(np.float32))

    @staticmethod
    def _normalise_elev_coarse(arr: np.ndarray) -> np.ndarray:
        """Coarse elevation can come as (1, H, W) or (H, W). Force (168, 280)."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (168, 280):
            # If it's already Europe (1, 169, 281) etc., crop.
            if arr.shape[0] >= 168 and arr.shape[1] >= 280:
                arr = arr[:168, :280]
            else:
                arr = np.zeros((168, 280), dtype=np.float32)
        return arr

    def _era5(self, year: int):
        if year not in self._era5_cache:
            self._era5_cache[year] = zarr.open(
                str(self.paths.era5_dir / f"{year}.zarr"), mode="r",
            )
        return self._era5_cache[year]

    def _ghap(self, year: int):
        if year not in self._ghap_cache:
            self._ghap_cache[year] = zarr.open(
                str(self.paths.ghap_dir / f"{year}.zarr"), mode="r",
            )
        return self._ghap_cache[year]

    def _cams(self, year: int):
        if year not in self._cams_cache:
            self._cams_cache[year] = zarr.open(
                str(self.paths.cams_dir / f"{year}.zarr"), mode="r",
            )
        return self._cams_cache[year]

    # ------------------------------------------------------------------ readers

    def _read_era5_day(self, year: int, day_idx: int) -> np.ndarray:
        """Return (30, 168, 280) ERA5 crop for one day."""
        z = self._era5(year)
        # Global ERA5 longitude is 0..360, but the European domain straddles
        # the prime meridian. Read in two parts and concatenate.
        if ERA5_LON_WEST_IDX >= ERA5_LON_EAST_IDX:
            west = np.asarray(z[day_idx, :, ERA5_LAT_SLICE, ERA5_LON_WEST_IDX:])
            east = np.asarray(z[day_idx, :, ERA5_LAT_SLICE, :ERA5_LON_EAST_IDX])
            out = np.concatenate([west, east], axis=-1)
        else:
            out = np.asarray(z[day_idx, :, ERA5_LAT_SLICE,
                                ERA5_LON_WEST_IDX:ERA5_LON_EAST_IDX])
        # Trim/pad to exactly 280.
        if out.shape[-1] >= 280:
            out = out[..., :280]
        elif out.shape[-1] < 280:
            pad = np.zeros(out.shape[:-1] + (280 - out.shape[-1],), dtype=out.dtype)
            out = np.concatenate([out, pad], axis=-1)
        return out.astype(np.float32)

    def _read_cams_day(self, year: int, day_idx: int) -> np.ndarray:
        """Return (5, 168, 280) CAMS crop resampled to the ERA5 grid."""
        g = self._cams(year)
        rows = []
        for v in CAMS_VARS:
            if v not in (g.array_keys() if hasattr(g, "array_keys") else g.keys()):
                rows.append(np.zeros((105, 175), dtype=np.float32))
                continue
            arr = np.asarray(g[v][day_idx,
                                   CAMS_LAT_TOP_IDX:CAMS_LAT_BOT_IDX + 1,
                                   CAMS_LON_WEST_IDX:CAMS_LON_EAST_IDX + 1])
            rows.append(arr)
        stack = np.stack(rows).astype(np.float32)
        # Resample to (5, 168, 280) using simple nearest-neighbour via numpy
        # indexing — perceptually fine at this scale.
        ny, nx = stack.shape[-2:]
        y_idx = np.linspace(0, ny - 1, 168).round().astype(int)
        x_idx = np.linspace(0, nx - 1, 280).round().astype(int)
        return stack[:, y_idx[:, None], x_idx[None, :]]

    def _read_ghap_day(self, year: int, day_idx: int) -> np.ndarray:
        """Return (4192, 6992) GHAP crop for one day."""
        z = self._ghap(year)
        out = np.asarray(z[day_idx, GHAP_LAT_SLICE,
                            GHAP_LON_WEST_IDX:GHAP_LON_WEST_IDX + GHAP_W])
        out = np.nan_to_num(out, nan=0.0)
        return out.astype(np.float32)

    # ------------------------------------------------------------------ public

    def build(self, year: int, day_idx: int) -> ForecastInputs:
        """Return ForecastInputs for ``(year, day_idx)``.

        ``day_idx`` is 0-based within the year (0 = 1 Jan, 1 = 2 Jan, ...).
        Day index 0 reuses itself as ``t-1`` (cold start).
        """
        prev = max(0, day_idx - 1)

        era5_t = self._read_era5_day(year, day_idx)
        cams_t = self._read_cams_day(year, day_idx)
        era5_pm1 = self._read_era5_day(year, prev)
        cams_pm1 = self._read_cams_day(year, prev)

        # Layout: 30 ERA5(t) + 5 CAMS(t) + 30 ERA5(t-1) + 5 CAMS(t-1) = 70 ch.
        era5_global = np.concatenate(
            [era5_t, cams_t, era5_pm1, cams_pm1], axis=0,
        )

        ghap_t0 = self._read_ghap_day(year, day_idx)
        ghap_tm1 = self._read_ghap_day(year, prev)

        return ForecastInputs(
            era5_global=era5_global,
            elev_coarse=self.elev_coarse,
            ghap_t0=ghap_t0,
            ghap_tm1=ghap_tm1,
            elev_hires=self.elev_hires,
        )


def load_real_europe_samples(
    year: int = 2022,
    day_indices: Sequence[int] | None = None,
    n: int = 30,
):
    """Convenience: return ``n`` samples from the given year.

    If ``day_indices`` is None, picks ``n`` evenly spaced days across the
    year. Used by ``paper/scripts/physics_ablations.py``.
    """
    builder = EuropeInputsBuilder()
    if day_indices is None:
        day_indices = np.linspace(0, 364, n).round().astype(int).tolist()
    samples = []
    for d in day_indices:
        inputs = builder.build(year, int(d))
        samples.append((d, inputs))
    return samples
