"""Tests for the public inference API: tiling, normalisation, checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cranpm.inference import iter_tiles, n_tiles, stitch_tiles
from cranpm.inference.checkpoint import Checkpoint
from cranpm.inference.normalize import (
    CAMS_MEANS,
    CAMS_STDS,
    ERA5_MEANS,
    ERA5_STDS,
    N_CAMS,
    N_ERA5,
)
from cranpm.inference.tiling import GHAP_H, GHAP_W, TILE_SIZE


def test_normalisation_constants_shape():
    assert ERA5_MEANS.shape == (N_ERA5,)
    assert ERA5_STDS.shape == (N_ERA5,)
    assert CAMS_MEANS.shape == (N_CAMS,)
    assert CAMS_STDS.shape == (N_CAMS,)
    assert (ERA5_STDS > 0).all()
    assert (CAMS_STDS > 0).all()


def test_iter_tiles_covers_full_grid():
    tiles = list(iter_tiles())
    assert len(tiles) == n_tiles()
    # Last tile must reach the bottom-right corner.
    assert any(t.row + t.size == GHAP_H for t in tiles)
    assert any(t.col + t.size == GHAP_W for t in tiles)
    # Every pixel must be covered by at least one tile.
    cover = np.zeros((GHAP_H, GHAP_W), dtype=bool)
    for t in tiles:
        cover[t.row:t.row + t.size, t.col:t.col + t.size] = True
    assert cover.all()


def test_stitch_tiles_recovers_constant():
    tiles = list(iter_tiles())
    payload = [(t, np.full((t.size, t.size), 7.5, dtype=np.float32)) for t in tiles]
    out = stitch_tiles(payload)
    assert out.shape == (GHAP_H, GHAP_W)
    assert np.allclose(out, 7.5, atol=1e-3)


def test_stitch_tiles_recovers_linear():
    """A linear lat gradient should be reconstructed by overlap-averaging."""
    tiles = list(iter_tiles())
    payload = []
    for t in tiles:
        ramp_y = np.arange(t.row, t.row + t.size, dtype=np.float32)
        arr = np.broadcast_to(ramp_y[:, None], (t.size, t.size)).astype(np.float32)
        payload.append((t, arr))
    out = stitch_tiles(payload)
    expected = np.arange(GHAP_H, dtype=np.float32)[:, None]
    diff = np.abs(out - expected).max()
    assert diff < 1.0, f"max diff {diff}"


def test_checkpoint_roundtrip_safetensors(tmp_path: Path):
    import torch

    state = {
        "global_branch.weight": torch.randn(4, 4),
        "local_branch.weight": torch.randn(2, 2),
    }
    config = {
        "model": {"era5_channels": 70, "global_img_size": [168, 280]},
        "version": "0.1.0-test",
    }
    ck = Checkpoint(config=config, state_dict=state, source=tmp_path)
    ck.to_safetensors_dir(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()

    reloaded = Checkpoint.from_path(tmp_path)
    assert reloaded.config == config
    assert set(reloaded.state_dict.keys()) == set(state.keys())
    for k in state:
        assert torch.equal(reloaded.state_dict[k], state[k])


def test_forecaster_from_pretrained_local(tmp_path: Path):
    """End-to-end: build a tiny model, save it, reload via from_pretrained."""
    import torch

    from cranpm import CRANPMForecaster, MultiScaleTopoFlow

    model_cfg = {
        "era5_channels": 70,
        "global_img_size": [168, 280],
        "global_patch_size": 8,
        "global_embed_dim": 192,
        "global_depth": 2,
        "global_num_heads": 4,
        "local_channels": 5,
        "local_img_size": [128, 128],
        "local_patch_size": 16,
        "local_embed_dim": 128,
        "local_depth": 2,
        "local_num_heads": 4,
        "cross_num_heads": 4,
        "cross_layers": 1,
        "decoder_depth": 1,
        "out_channels": 1,
        "mlp_ratio": 4.0,
        "drop_rate": 0.0,
        "drop_path": 0.0,
        "global_region_h": 7,
        "global_region_w": 7,
    }
    model = MultiScaleTopoFlow(**{k: tuple(v) if k.endswith("img_size") else v
                                  for k, v in model_cfg.items()})
    ck = Checkpoint(config={"model": model_cfg}, state_dict=model.state_dict(),
                    source=tmp_path)
    ck.to_safetensors_dir(tmp_path)

    fc = CRANPMForecaster.from_pretrained(tmp_path, device="cpu")
    assert fc.model is not None
    assert fc.config["model"]["era5_channels"] == 70


@pytest.mark.parametrize("c", [35, 70])
def test_normalise_era5_accepts_35_or_70(c):
    from cranpm.inference.forecaster import CRANPMForecaster

    arr = np.zeros((c, 168, 280), dtype=np.float32)
    fc = object.__new__(CRANPMForecaster)
    out = fc._normalise_era5(arr)
    assert out.shape == (70, 168, 280)
