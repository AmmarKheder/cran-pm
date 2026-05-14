"""Tile / un-tile helpers for inference on the full European grid.

The CRAN-PM local branch processes 512x512 tiles. A full pan-European
forecast covers a 4192 x 6992 grid (GHAP). We tile the domain with
overlap (default stride 384) to mitigate boundary artefacts, then
average the overlapping regions on stitch-back.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# GHAP Europe grid (defined in cranpm.data.europe_dataset and matched here).
GHAP_H = 4192
GHAP_W = 6992
TILE_SIZE = 512
DEFAULT_STRIDE = 384  # 75% of tile -> ~25% overlap


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    size: int = TILE_SIZE


def iter_tiles(
    h: int = GHAP_H, w: int = GHAP_W, size: int = TILE_SIZE, stride: int = DEFAULT_STRIDE
) -> Iterator[Tile]:
    """Iterate top-left coordinates of overlapping tiles covering (h, w).

    The last row / column of tiles is anchored to the right edge so the
    full grid is covered without padding.
    """
    rows = list(range(0, h - size + 1, stride))
    if rows[-1] != h - size:
        rows.append(h - size)
    cols = list(range(0, w - size + 1, stride))
    if cols[-1] != w - size:
        cols.append(w - size)
    for r in rows:
        for c in cols:
            yield Tile(row=r, col=c, size=size)


def stitch_tiles(
    tiles: list[tuple[Tile, np.ndarray]],
    h: int = GHAP_H,
    w: int = GHAP_W,
) -> np.ndarray:
    """Stitch tile predictions back into a (h, w) array, averaging overlaps."""
    accum = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for tile, arr in tiles:
        s = tile.size
        # Raised cosine (Hann-like) tapered weight, with a floor to avoid
        # zero contribution at the very corners of the GHAP grid (where
        # only a single tile covers that pixel).
        ramp = 0.5 + 0.5 * np.cos(np.linspace(-np.pi, np.pi, s))
        ramp = np.maximum(ramp, 0.05)
        w2d = np.outer(ramp, ramp).astype(np.float32)
        accum[tile.row:tile.row + s, tile.col:tile.col + s] += arr.astype(np.float32) * w2d
        weight[tile.row:tile.row + s, tile.col:tile.col + s] += w2d
    out = np.where(weight > 0, accum / np.maximum(weight, 1e-8), 0.0)
    return out


def n_tiles(h: int = GHAP_H, w: int = GHAP_W, size: int = TILE_SIZE, stride: int = DEFAULT_STRIDE) -> int:
    return sum(1 for _ in iter_tiles(h, w, size, stride))
