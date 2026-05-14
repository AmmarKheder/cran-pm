"""High-level forecasting API for CRAN-PM.

The :class:`CRANPMForecaster` is the main entry point for end users.
It hides the multi-branch model assembly, the data normalisation, the
tile-based inference and the de-normalisation behind two methods:

>>> model = CRANPMForecaster.from_pretrained("AmmarKheder/cran-pm-europe-v3")
>>> forecast = model.predict(era5=..., cams=..., elevation=..., ghap_t0=...,
...                           lead_time=1)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cranpm.config import get_default_device
from cranpm.inference.checkpoint import Checkpoint, from_huggingface_hub
from cranpm.inference.normalize import (
    CAMS_MEANS,
    CAMS_STDS,
    ELEV_MEAN,
    ELEV_STD,
    ERA5_MEANS,
    ERA5_STDS,
    GHAP_MEAN,
    GHAP_STD,
    N_CAMS,
    N_ERA5,
)
from cranpm.inference.tiling import GHAP_H, GHAP_W, Tile, iter_tiles, stitch_tiles
from cranpm.models.multiscale_topoflow import MultiScaleTopoFlow

LAT_NORTH = 72.0
LON_WEST = -25.0
GHAP_RES_DEG = 0.01


@dataclass
class ForecastInputs:
    """Pre-stacked input arrays used by :meth:`CRANPMForecaster.predict`.

    Shapes (numpy float32):
        era5_global: (70, 168, 280)   -- 30+5 channels at t and t-1
        elev_coarse: (168, 280)       -- elevation on the global grid
        ghap_t0:     (4192, 6992)     -- PM2.5 today, raw units
        ghap_tm1:    (4192, 6992)     -- PM2.5 yesterday, raw units
        elev_hires:  (4192, 6992)     -- elevation at the GHAP grid
    """

    era5_global: np.ndarray
    elev_coarse: np.ndarray
    ghap_t0: np.ndarray
    ghap_tm1: np.ndarray
    elev_hires: np.ndarray


class CRANPMForecaster:
    """End-user forecasting API."""

    def __init__(
        self,
        model: MultiScaleTopoFlow,
        config: dict[str, Any],
        device: str | torch.device = "cpu",
        precision: str = "fp32",
    ):
        self.model = model.eval().to(device)
        self.config = config
        self.device = torch.device(device)
        self.precision = precision
        self._dtype_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        if precision not in self._dtype_map:
            raise ValueError(f"unknown precision {precision!r}")

    # ----------------------------------------------------------------- factories

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str | Path,
        device: str | None = None,
        precision: str = "fp32",
        token: str | None = None,
    ) -> "CRANPMForecaster":
        """Load a CRAN-PM model from the HuggingFace Hub or a local path.

        ``repo_or_path`` may be:

        * an HF Hub identifier such as ``"AmmarKheder/cran-pm-europe-v3"``,
        * an absolute or relative path to a directory containing
          ``model.safetensors`` + ``config.json``,
        * a path to a Lightning ``.ckpt`` file.
        """
        device = device or get_default_device()

        path = Path(str(repo_or_path))
        if path.exists():
            ckpt = Checkpoint.from_path(path)
        else:
            ckpt = from_huggingface_hub(str(repo_or_path), token=token)

        model_cfg = ckpt.config.get("model", ckpt.config)
        model = MultiScaleTopoFlow(**_filter_model_kwargs(model_cfg))
        missing, unexpected = model.load_state_dict(ckpt.state_dict, strict=False)
        if missing:
            print(f"warning: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"warning: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        return cls(model=model, config=ckpt.config, device=device, precision=precision)

    # ----------------------------------------------------------------- inference

    @torch.no_grad()
    def predict(
        self,
        inputs: ForecastInputs,
        lead_time: int = 1,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> np.ndarray:
        """Run a full pan-European forecast at 1 km.

        Returns a (4192, 6992) numpy array of PM2.5 in raw units (µg/m³).
        """
        # 1. Normalise inputs.
        era5 = self._normalise_era5(inputs.era5_global)
        elev_c = (inputs.elev_coarse - ELEV_MEAN) / ELEV_STD
        ghap_norm = (inputs.ghap_t0 - GHAP_MEAN) / GHAP_STD
        ghap_prev_norm = (inputs.ghap_tm1 - GHAP_MEAN) / GHAP_STD
        elev_h_norm = (inputs.elev_hires - ELEV_MEAN) / ELEV_STD

        # 2. Pre-build coordinate maps on the GHAP grid.
        rows = np.arange(GHAP_H, dtype=np.float32)
        cols = np.arange(GHAP_W, dtype=np.float32)
        lats_grid = LAT_NORTH - rows * GHAP_RES_DEG
        lons_grid = LON_WEST + cols * GHAP_RES_DEG

        # Common tensors for the global branch (shared across tiles).
        era5_t = torch.from_numpy(era5).unsqueeze(0).to(self.device)
        elev_c_t = torch.from_numpy(elev_c).unsqueeze(0).to(self.device)

        # 3. Tile loop.
        tiles = list(iter_tiles())
        results: list[tuple[Tile, np.ndarray]] = []
        for batch in _batched(tiles, batch_size):
            local_in = []
            elev_h_in = []
            patch_centers = []
            for tile in batch:
                local_in.append(self._build_local_patch(
                    tile, ghap_norm, ghap_prev_norm, elev_h_norm,
                    lats_grid, lons_grid,
                ))
                elev_h_in.append(elev_h_norm[
                    tile.row:tile.row + tile.size,
                    tile.col:tile.col + tile.size,
                ])
                # Patch centre in normalised coords (lat, lon -> [-1, 1]).
                cy = tile.row + tile.size / 2
                cx = tile.col + tile.size / 2
                pc_lat = (LAT_NORTH - cy * GHAP_RES_DEG - 50) / 25.0
                pc_lon = (LON_WEST + cx * GHAP_RES_DEG - 10) / 35.0
                patch_centers.append([pc_lat, pc_lon])

            local_t = torch.from_numpy(np.stack(local_in)).to(self.device)
            elev_h_t = torch.from_numpy(np.stack(elev_h_in)).to(self.device)
            pc_t = torch.tensor(patch_centers, device=self.device)
            era5_b = era5_t.expand(len(batch), -1, -1, -1)
            elev_c_b = elev_c_t.expand(len(batch), -1, -1)
            lt_t = torch.full((len(batch),), float(lead_time), device=self.device)

            with torch.autocast(device_type=self.device.type,
                                 dtype=self._dtype_map[self.precision],
                                 enabled=self.precision != "fp32"):
                pred = self.model(
                    era5=era5_b,
                    elevation_coarse=elev_c_b,
                    ghap_patch=local_t,
                    elevation_hires=elev_h_t,
                    lead_time=lt_t,
                    patch_center=pc_t,
                    wind_at_patch=None,
                )
            # pred is (B, 1, 512, 512), normalised
            pred_np = pred.float().cpu().numpy()[:, 0]
            for tile, p in zip(batch, pred_np):
                results.append((tile, p * GHAP_STD + GHAP_MEAN))

            if verbose:
                done = len(results)
                print(f"  {done}/{len(tiles)} tiles done")

        return stitch_tiles(results)

    # ----------------------------------------------------------------- helpers

    def _normalise_era5(self, era5: np.ndarray) -> np.ndarray:
        """Apply per-channel normalisation. Accepts (C, H, W) with C in {35, 70}.

        For C=35, the input is treated as a single time step and the model
        receives it duplicated as t and t-1.
        """
        c = era5.shape[0]
        if c not in (35, 70):
            raise ValueError(f"era5 must have 35 or 70 channels, got {c}")
        if c == 35:
            era5 = np.concatenate([era5, era5], axis=0)
        # First 30 channels = ERA5, next 5 = CAMS.
        norm = era5.astype(np.float32).copy()
        for off in (0, 35):
            norm[off:off + N_ERA5] = (
                norm[off:off + N_ERA5] - ERA5_MEANS[:, None, None]
            ) / ERA5_STDS[:, None, None]
            norm[off + N_ERA5:off + N_ERA5 + N_CAMS] = (
                norm[off + N_ERA5:off + N_ERA5 + N_CAMS]
                - CAMS_MEANS[:, None, None]
            ) / CAMS_STDS[:, None, None]
        return norm

    def _build_local_patch(
        self,
        tile: Tile,
        ghap_norm: np.ndarray,
        ghap_prev_norm: np.ndarray,
        elev_h_norm: np.ndarray,
        lats_grid: np.ndarray,
        lons_grid: np.ndarray,
    ) -> np.ndarray:
        s = tile.size
        ghap_p = ghap_norm[tile.row:tile.row + s, tile.col:tile.col + s]
        ghap_pm1 = ghap_prev_norm[tile.row:tile.row + s, tile.col:tile.col + s]
        elev_p = elev_h_norm[tile.row:tile.row + s, tile.col:tile.col + s]
        lats_p = np.broadcast_to(
            lats_grid[tile.row:tile.row + s, None] / 90.0, (s, s)
        )
        lons_p = np.broadcast_to(
            lons_grid[None, tile.col:tile.col + s] / 180.0, (s, s)
        )
        return np.stack([ghap_p, elev_p, lats_p, lons_p, ghap_pm1]).astype(np.float32)


def _batched(it: Iterable, n: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


_MODEL_KWARG_NAMES = {
    "era5_channels", "global_img_size", "global_patch_size",
    "global_embed_dim", "global_depth", "global_num_heads",
    "local_channels", "local_img_size", "local_patch_size",
    "local_embed_dim", "local_depth", "local_num_heads",
    "cross_num_heads", "cross_layers",
    "decoder_depth", "out_channels",
    "mlp_ratio", "drop_rate", "drop_path",
    "global_region_h", "global_region_w",
}


def _filter_model_kwargs(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if k in _MODEL_KWARG_NAMES:
            if k.endswith("_img_size") and isinstance(v, list):
                v = tuple(v)
            out[k] = v
    return out
