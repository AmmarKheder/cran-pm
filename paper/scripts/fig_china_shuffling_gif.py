#!/usr/bin/env python3
"""Animated visualization of patch-shuffling strategies on the China domain.

Three traversal orderings are compared:

  1. Random         (independent permutation)
  2. Raster         (row-major, serpentine = boustrophedon)
  3. Wind-guided    (sorted by projection onto the daily-mean wind vector)

For each strategy the grid is filled tile-by-tile in traversal order,
showing the *path* the transformer would take through the input tokens.
The accompanying caption follows the framing of REOrder (Goyal et al.
2024, https://d3tk.github.io/REOrder/): the ordering acts as a learned
inductive bias and the right traversal accelerates training.

Output: paper/figures/fig_china_shuffling_paths.{gif,png}

The script uses pre-cached ERA5/GHAP global zarrs; no GPU is required.
Run on a CPU node with matplotlib + Pillow installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
import zarr


# ── Geographic window ─────────────────────────────────────────────────────────
# 42°×70° "China" window matching the OOD inference layout.
ERA5_LAT_NORTH = 50.0
ERA5_LON_WEST = 75.0
ERA5_RES = 0.25
ERA5_H, ERA5_W = 169, 281

GHAP_LAT_NORTH = 45.0
GHAP_LON_WEST = 80.0
GHAP_RES = 0.01
GHAP_H, GHAP_W = 4192, 6992

# Tile grid (matches the 32×32 ablation in §S3.7 of the screenshot).
N_ROWS = 32
N_COLS = 32

DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")


def crop_era5(day, lat_n, lon_w):
    z = zarr.open(str(DATA / "era5_global_daily" / "2022.zarr"), mode="r",
                  zarr_format=2)
    lat0 = int(round((90.0 - lat_n) / ERA5_RES))
    lon0_360 = int(round((lon_w % 360) / ERA5_RES))
    arr = z[day, :, lat0:lat0 + ERA5_H,
            lon0_360:lon0_360 + ERA5_W]
    return np.asarray(arr, dtype=np.float32)


def crop_ghap_thumb(day, lat_n, lon_w, down=10):
    """Cropped GHAP at native 0.01° but down-sampled by ``down`` for plotting."""
    z = zarr.open(str(DATA / "ghap_global_daily" / "2022.zarr"), mode="r",
                  zarr_format=2)
    lat0 = int(round((90.0 - lat_n) / GHAP_RES))
    lon0 = int(round((lon_w - (-180.0)) / GHAP_RES))
    arr = z[day, lat0:lat0 + GHAP_H:down, lon0:lon0 + GHAP_W:down]
    return np.asarray(arr, dtype=np.float32)


# ── Traversal orderings ──────────────────────────────────────────────────────
def order_random(n_rows, n_cols, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows * n_cols)
    rng.shuffle(idx)
    return [(int(i // n_cols), int(i % n_cols)) for i in idx]


def order_raster_serpentine(n_rows, n_cols):
    """Row-major serpentine (boustrophedon) order."""
    out = []
    for r in range(n_rows):
        cs = range(n_cols) if r % 2 == 0 else range(n_cols - 1, -1, -1)
        for c in cs:
            out.append((r, c))
    return out


def order_wind_guided(n_rows, n_cols, wind_u, wind_v):
    """Sort tiles by projection onto the mean wind vector.

    Direction is the standard meteorological "from where the wind blows
    *to*" — large projection = downstream.
    """
    # Use the median wind to ignore sub-grid gusts.
    u = float(np.nanmedian(wind_u))
    v = float(np.nanmedian(wind_v))
    norm = np.hypot(u, v) + 1e-6
    u /= norm
    v /= norm
    # Tile centre coordinates in [0,1]×[0,1] (row 0 = north, col 0 = west).
    rows = (np.arange(n_rows) + 0.5) / n_rows
    cols = (np.arange(n_cols) + 0.5) / n_cols
    # Lon increases west→east (positive u: tiles to the east downstream).
    # Lat decreases north→south, so positive v (northerly wind component
    # in the meteorological convention is "from the south") means south
    # tiles are upstream → traverse rows in reverse.
    proj = np.zeros((n_rows, n_cols), dtype=np.float64)
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            # x = lon, y = -lat (so larger y = south)
            proj[ri, ci] = u * c + v * (1.0 - r)
    flat_idx = np.argsort(proj.ravel())
    return [(int(i // n_cols), int(i % n_cols)) for i in flat_idx]


# ── Tile aggregation for visualization ───────────────────────────────────────
def aggregate_tile_pm25(ghap_thumb):
    """Down-sample the GHAP thumbnail to the tile grid using mean pooling."""
    h, w = ghap_thumb.shape
    tile_h = h // N_ROWS
    tile_w = w // N_COLS
    cropped = ghap_thumb[:N_ROWS * tile_h, :N_COLS * tile_w]
    return cropped.reshape(N_ROWS, tile_h, N_COLS, tile_w).mean(axis=(1, 3))


def aggregate_tile_wind(era5):
    """Mean (u, v) per tile from the 169×281 ERA5 surface fields."""
    u10 = era5[0]
    v10 = era5[1]
    rs = np.linspace(0, u10.shape[0], N_ROWS + 1, dtype=int)
    cs = np.linspace(0, u10.shape[1], N_COLS + 1, dtype=int)
    u_t = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    v_t = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    for i in range(N_ROWS):
        for j in range(N_COLS):
            u_t[i, j] = u10[rs[i]:rs[i+1], cs[j]:cs[j+1]].mean()
            v_t[i, j] = v10[rs[i]:rs[i+1], cs[j]:cs[j+1]].mean()
    return u_t, v_t


# ── Animation ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, default=20,
                        help="Day-of-year (0-364) to use for the wind field.")
    parser.add_argument("--output", type=Path,
                        default=Path("/scratch/project_462001140/ammar/eccv/"
                                     "cran-pm/paper/figures/"
                                     "fig_china_shuffling_paths"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames-per-strategy", type=int, default=64,
                        help="Tiles drawn per frame ~ N_ROWS*N_COLS / this.")
    args = parser.parse_args()

    print(f"Loading ERA5 day={args.day} ...", flush=True)
    era5 = crop_era5(args.day, ERA5_LAT_NORTH, ERA5_LON_WEST)
    print(f"Loading GHAP day={args.day} ...", flush=True)
    ghap_thumb = crop_ghap_thumb(args.day, GHAP_LAT_NORTH, GHAP_LON_WEST,
                                  down=10)  # 419×699
    print(f"  era5={era5.shape}, ghap_thumb={ghap_thumb.shape}", flush=True)

    pm_tile = aggregate_tile_pm25(ghap_thumb)
    u_tile, v_tile = aggregate_tile_wind(era5)

    strategies = [
        ("Random",        order_random(N_ROWS, N_COLS, seed=42)),
        ("Raster (serpentine)", order_raster_serpentine(N_ROWS, N_COLS)),
        ("Wind-guided (CRAN-PM)", order_wind_guided(N_ROWS, N_COLS,
                                                     u_tile, v_tile)),
    ]
    strategy_colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    tiles_per_frame = max(1, (N_ROWS * N_COLS) // args.frames_per_strategy)
    n_frames_per_strat = (N_ROWS * N_COLS) // tiles_per_frame + 4

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2),
                              gridspec_kw={"wspace": 0.18,
                                           "width_ratios": [1.05, 1.05, 1.05]})
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap("RdYlBu_r")
    vmin = float(np.nanpercentile(pm_tile, 5))
    vmax = float(np.nanpercentile(pm_tile, 95))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Base background — same for all three panels.
    extent = (GHAP_LON_WEST, GHAP_LON_WEST + GHAP_W * GHAP_RES,
              GHAP_LAT_NORTH - GHAP_H * GHAP_RES, GHAP_LAT_NORTH)

    panels = []
    for k, (name, order) in enumerate(strategies):
        ax = axes[k]
        ax.set_facecolor("#f5f5f5")
        ax.imshow(np.full_like(pm_tile, np.nan), extent=extent,
                  origin="upper", cmap=cmap, norm=norm,
                  interpolation="nearest", alpha=0.0)
        # Quiver of wind (sub-sampled).
        lons = np.linspace(GHAP_LON_WEST, GHAP_LON_WEST + GHAP_W * GHAP_RES,
                           N_COLS, endpoint=False) + GHAP_RES * GHAP_W / (2 * N_COLS)
        lats = (np.linspace(GHAP_LAT_NORTH, GHAP_LAT_NORTH - GHAP_H * GHAP_RES,
                            N_ROWS, endpoint=False)
                - GHAP_RES * GHAP_H / (2 * N_ROWS))
        L, La = np.meshgrid(lons, lats)
        ax.quiver(L[::2, ::2], La[::2, ::2],
                  u_tile[::2, ::2], v_tile[::2, ::2],
                  scale=300, width=0.0026, color="#444444", alpha=0.55)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("auto")
        ax.set_title(f"{name}", fontsize=12, fontweight="bold",
                     color=strategy_colors[k])
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        if k == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.tick_params(labelsize=8)

        # Pre-create rectangles for every tile; we'll set their alpha/color
        # in the animator.
        rects = []
        tile_lon = (GHAP_W * GHAP_RES) / N_COLS
        tile_lat = (GHAP_H * GHAP_RES) / N_ROWS
        for r in range(N_ROWS):
            for c in range(N_COLS):
                x = GHAP_LON_WEST + c * tile_lon
                y = GHAP_LAT_NORTH - (r + 1) * tile_lat
                rect = mpatches.Rectangle(
                    (x, y), tile_lon, tile_lat,
                    facecolor=cmap(norm(pm_tile[r, c])),
                    edgecolor="white", linewidth=0.25, alpha=0.0,
                    zorder=2,
                )
                ax.add_patch(rect)
                rects.append(rect)

        # The growing traversal polyline.
        path_line, = ax.plot([], [], color=strategy_colors[k],
                              linewidth=0.9, alpha=0.55, zorder=3,
                              solid_capstyle="round")
        # Marker for the current tile (head of the traversal).
        head_marker, = ax.plot([], [], "o", color=strategy_colors[k],
                                markersize=6,
                                markeredgecolor="white", markeredgewidth=0.8,
                                zorder=4)
        panels.append({"ax": ax, "rects": rects, "path_line": path_line,
                       "head_marker": head_marker, "order": order})

    cax = fig.add_axes([0.265, 0.06, 0.5, 0.018])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=cax, orientation="horizontal")
    cb.set_label("Daily-mean PM$_{2.5}$ (µg m$^{-3}$) — colour-coded "
                  "by ground-truth value, the model only sees the order",
                  fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Patch traversal strategies on the China OOD domain "
        f"(day-of-year {args.day}, 2022)",
        fontsize=13.5, y=0.97,
    )

    def init():
        artists = []
        for p in panels:
            for r in p["rects"]:
                r.set_alpha(0.0)
            p["path_line"].set_data([], [])
            p["head_marker"].set_data([], [])
            artists.extend(p["rects"])
            artists.append(p["path_line"])
            artists.append(p["head_marker"])
        return artists

    def step(frame):
        artists = []
        n_visible = min(N_ROWS * N_COLS, (frame + 1) * tiles_per_frame)
        # Fade the path tail so the *head* of the traversal stays salient
        # even at the end (avoids the dense-spaghetti final frame).
        tail_len = max(64, N_ROWS * N_COLS // 6)
        for p in panels:
            for r in p["rects"]:
                r.set_alpha(0.0)
            xs, ys = [], []
            tile_lon = (GHAP_W * GHAP_RES) / N_COLS
            tile_lat = (GHAP_H * GHAP_RES) / N_ROWS
            for k in range(n_visible):
                r, c = p["order"][k]
                idx = r * N_COLS + c
                p["rects"][idx].set_alpha(0.95)
                xs.append(GHAP_LON_WEST + (c + 0.5) * tile_lon)
                ys.append(GHAP_LAT_NORTH - (r + 0.5) * tile_lat)
            tail_start = max(0, len(xs) - tail_len)
            p["path_line"].set_data(xs[tail_start:], ys[tail_start:])
            if xs:
                p["head_marker"].set_data([xs[-1]], [ys[-1]])
            else:
                p["head_marker"].set_data([], [])
            artists.extend(p["rects"])
            artists.append(p["path_line"])
            artists.append(p["head_marker"])
        return artists

    print("Rendering animation ...", flush=True)
    anim = animation.FuncAnimation(
        fig, step, init_func=init,
        frames=n_frames_per_strat, interval=1000.0 / args.fps,
        blit=False, repeat=False,
    )

    out_gif = args.output.with_suffix(".gif")
    out_png = args.output.with_suffix(".png")
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=args.fps)
    anim.save(str(out_gif), writer=writer, dpi=110)
    print(f"Wrote {out_gif}")

    # Save a "mid-traversal" frame as the static PNG so the path is still
    # readable (a fully populated grid + full polyline is illegible).
    step(int(n_frames_per_strat * 0.45))
    fig.savefig(str(out_png), dpi=180, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
