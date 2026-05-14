#!/usr/bin/env python3
"""Two static figures arguing the *theoretical* role of patch ordering.

  fig_shuffle_permutation_matrix.{pdf,png}
    3-panel heatmap showing the permutation matrix Pi for random /
    raster / wind-guided. Bandwidth (max |i - sigma(i)|) annotated.

  fig_shuffle_lagrangian_frame.{pdf,png}
    Each tile coloured by its position in the traversal sequence,
    overlaid with the ERA5 wind quiver. Wind-guided makes the
    sequence-index field nearly *parallel* to u; raster makes it
    parallel to lon/lat axes; random has no structure.

These complement fig_china_shuffling_paths.gif (the animation): they
make the same point but in a publication-friendly static form.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import zarr


# Geographic window — identical to the GIF script.
ERA5_LAT_NORTH = 50.0
ERA5_LON_WEST = 75.0
ERA5_RES = 0.25
ERA5_H, ERA5_W = 169, 281

GHAP_LAT_NORTH = 45.0
GHAP_LON_WEST = 80.0
GHAP_RES = 0.01
GHAP_H, GHAP_W = 4192, 6992

N_ROWS = 32
N_COLS = 32

DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")


def crop_era5(day):
    z = zarr.open(str(DATA / "era5_global_daily" / "2022.zarr"), mode="r",
                  zarr_format=2)
    lat0 = int(round((90.0 - ERA5_LAT_NORTH) / ERA5_RES))
    lon0 = int(round((ERA5_LON_WEST % 360) / ERA5_RES))
    return np.asarray(z[day, :, lat0:lat0 + ERA5_H,
                         lon0:lon0 + ERA5_W], dtype=np.float32)


def aggregate_tile_wind(era5):
    u10, v10 = era5[0], era5[1]
    rs = np.linspace(0, u10.shape[0], N_ROWS + 1, dtype=int)
    cs = np.linspace(0, u10.shape[1], N_COLS + 1, dtype=int)
    u_t = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    v_t = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    for i in range(N_ROWS):
        for j in range(N_COLS):
            u_t[i, j] = u10[rs[i]:rs[i+1], cs[j]:cs[j+1]].mean()
            v_t[i, j] = v10[rs[i]:rs[i+1], cs[j]:cs[j+1]].mean()
    return u_t, v_t


# ── traversal orderings (same as gif script) ───────────────────────────────
def order_random(n_rows, n_cols, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows * n_cols)
    rng.shuffle(idx)
    return idx


def order_raster_serpentine(n_rows, n_cols):
    out = []
    for r in range(n_rows):
        cs = range(n_cols) if r % 2 == 0 else range(n_cols - 1, -1, -1)
        for c in cs:
            out.append(r * n_cols + c)
    return np.asarray(out)


def order_wind_guided(n_rows, n_cols, u, v):
    """Sort tiles by projection onto the daily-mean wind vector."""
    um, vm = float(np.nanmedian(u)), float(np.nanmedian(v))
    norm = np.hypot(um, vm) + 1e-6
    um /= norm; vm /= norm
    proj = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        y = 1.0 - (i + 0.5) / n_rows           # north = high y
        for j in range(n_cols):
            x = (j + 0.5) / n_cols             # east = high x
            proj[i, j] = um * x + vm * y
    return np.argsort(proj.ravel())


# ── permutation matrix figure ──────────────────────────────────────────────
def permutation_matrix_figure(orders, names, colors, out_path):
    N = len(orders[0])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6),
                              gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor("white")

    for k, (order, name) in enumerate(zip(orders, names)):
        ax = axes[k]
        Pi = np.zeros((N, N), dtype=np.uint8)
        Pi[np.arange(N), order] = 1
        bandwidth = int(np.max(np.abs(np.arange(N) - order)))
        ax.imshow(Pi, cmap="RdYlBu_r", aspect="equal",
                  interpolation="nearest",
                  norm=mcolors.Normalize(vmin=0, vmax=1))
        ax.set_title(name, fontsize=12, fontweight="bold",
                     color=colors[k])
        ax.set_xlabel("target position $j$", fontsize=10)
        if k == 0:
            ax.set_ylabel("source tile $i$", fontsize=10)
        ax.text(0.97, 0.04,
                f"bandwidth\n$\\max|i-\\sigma(i)| = {bandwidth}$",
                transform=ax.transAxes, fontsize=9,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black",
                          alpha=0.9, boxstyle="round,pad=0.3"))
        ax.tick_params(labelsize=8)
    fig.suptitle(
        r"Permutation matrices $\Pi_{i,\sigma(i)}=1$ of three patch-ordering "
        r"strategies"
        "\n" r"Lower bandwidth $\Rightarrow$ better locality preservation; "
        r"a tilted diagonal $\Rightarrow$ structured (here wind-aligned) order",
        fontsize=12, y=1.02,
    )
    fig.savefig(str(out_path.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')} and .png")


# ── Lagrangian frame figure ───────────────────────────────────────────────
def lagrangian_frame_figure(orders, names, colors, u_tile, v_tile, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6),
                              gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor("white")

    # Tile centre coords for plotting (geographic).
    tile_lon = (GHAP_W * GHAP_RES) / N_COLS
    tile_lat = (GHAP_H * GHAP_RES) / N_ROWS
    lons = GHAP_LON_WEST + (np.arange(N_COLS) + 0.5) * tile_lon
    lats = GHAP_LAT_NORTH - (np.arange(N_ROWS) + 0.5) * tile_lat
    L, La = np.meshgrid(lons, lats)
    extent = (GHAP_LON_WEST, GHAP_LON_WEST + GHAP_W * GHAP_RES,
              GHAP_LAT_NORTH - GHAP_H * GHAP_RES, GHAP_LAT_NORTH)

    cmap = plt.get_cmap("RdYlBu_r")
    for k, (order, name) in enumerate(zip(orders, names)):
        ax = axes[k]
        pos_field = np.zeros(N_ROWS * N_COLS, dtype=np.float32)
        pos_field[order] = np.linspace(0.0, 1.0, N_ROWS * N_COLS)
        pos_grid = pos_field.reshape(N_ROWS, N_COLS)
        ax.imshow(pos_grid, extent=extent, origin="upper", cmap=cmap,
                  vmin=0.0, vmax=1.0, interpolation="nearest")
        # Wind overlay (sub-sampled).
        ax.quiver(L[::2, ::2], La[::2, ::2],
                  u_tile[::2, ::2], v_tile[::2, ::2],
                  scale=300, width=0.0026,
                  color="#222222", alpha=0.7)
        ax.set_title(name, fontsize=12, fontweight="bold",
                     color=colors[k])
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        if k == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.set_aspect("auto")
        ax.tick_params(labelsize=8)

    cax = fig.add_axes([0.28, 0.06, 0.45, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=0, vmax=1), cmap=cmap),
        cax=cax, orientation="horizontal")
    cb.set_label(
        r"Tile position in the transformer sequence ($i / N$)"
        " — wind quiver is the ERA5 daily-mean (u, v)",
        fontsize=10,
    )
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        r"Patch ordering as a discrete Lagrangian frame on the China OOD"
        " domain"
        "\n"
        r"Wind-guided $\Rightarrow$ sequence index field aligned with"
        r" $\mathbf{u}$, approximating the streamline coordinate of an"
        r" advective transport $\partial_t C + \mathbf{u}\cdot\nabla C = 0$",
        fontsize=12, y=1.02,
    )
    fig.savefig(str(out_path.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(out_path.with_suffix(".png")), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')} and .png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, default=20)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/scratch/project_462001140/ammar/eccv/"
                                     "cran-pm/paper/figures"))
    args = parser.parse_args()

    print(f"Loading ERA5 day={args.day} ...", flush=True)
    era5 = crop_era5(args.day)
    u_tile, v_tile = aggregate_tile_wind(era5)

    orders = [
        order_random(N_ROWS, N_COLS, seed=42),
        order_raster_serpentine(N_ROWS, N_COLS),
        order_wind_guided(N_ROWS, N_COLS, u_tile, v_tile),
    ]
    names = ["Random", "Raster (serpentine)", "Wind-guided (CRAN-PM)"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    permutation_matrix_figure(
        orders, names, colors,
        args.out_dir / "fig_shuffle_permutation_matrix",
    )
    lagrangian_frame_figure(
        orders, names, colors, u_tile, v_tile,
        args.out_dir / "fig_shuffle_lagrangian_frame",
    )


if __name__ == "__main__":
    main()
