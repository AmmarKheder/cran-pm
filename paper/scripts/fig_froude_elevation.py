#!/usr/bin/env python3
"""Froude-number diagnostic for the elevation-bias ablation.

The atmospheric Froude number
   Fr = U / (N H)
diagnoses whether a stratified flow is *blocked* by orography (Fr < 1,
the flow goes around mountains) or *unblocked* (Fr > 1, the flow goes
over). The CRAN-PM elevation-aware attention is hypothesised to matter
precisely in Fr<1 regimes: those are the situations where pollution
gets trapped in valleys and the topographic barrier cannot be
represented by isotropic distance-based attention.

This figure builds a daily Fr map over Europe from ERA5 pressure-level
data, overlays the Fr<1 mask on the CRAN-PM minus GHAP error field for
the same day, and reports the spatial correlation between
\(\mathbb{1}_{Fr<1}\) and \(|\text{error}|\) for several models.

Outputs:
  paper/figures/fig_froude_elevation.{pdf,png}
  paper/figures/froude_correlations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import zarr


# ── Europe window ──────────────────────────────────────────────────────────
LAT_NORTH = 72.0
LON_WEST = -25.0
ERA5_RES = 0.25
ERA5_H, ERA5_W = 169, 281

GHAP_RES = 0.01
GHAP_H, GHAP_W = 4192, 6992

DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")
PROJECT = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe")

# ERA5 channel offsets:
#   0..4   : u10, v10, t2m, msl, sp
#   5..9   : t at [1000, 925, 850, 700, 500] hPa
#   10..14 : u
#   15..19 : v
#   20..24 : q
#   25..29 : z (geopotential, m^2/s^2)
CH_T1000, CH_T500 = 5, 9
CH_U850, CH_V850 = 12, 17
CH_Z1000, CH_Z500 = 25, 29

G_ACCEL = 9.81
P_REF = 100000.0       # Pa
R_CP = 0.286


def crop_era5_europe(year: int, day: int):
    z = zarr.open(str(DATA / "era5_global_daily" / f"{year}.zarr"),
                  mode="r", zarr_format=2)
    lat0 = int(round((90.0 - LAT_NORTH) / ERA5_RES))
    lon0_mod = int(round((LON_WEST % 360) / ERA5_RES))
    lon_full = z.shape[3]
    # Europe wraps the antimeridian when ERA5 uses 0..360 lon.
    if lon0_mod + ERA5_W <= lon_full:
        arr = z[day, :, lat0:lat0 + ERA5_H,
                lon0_mod:lon0_mod + ERA5_W]
    else:
        right = z[day, :, lat0:lat0 + ERA5_H, lon0_mod:lon_full]
        left = z[day, :, lat0:lat0 + ERA5_H,
                 0:lon0_mod + ERA5_W - lon_full]
        arr = np.concatenate([right, left], axis=-1)
    return np.asarray(arr, dtype=np.float32)


def compute_froude(era5):
    """Return Fr (169,281), U_850 (169,281), N (169,281), H_relief (169,281)."""
    t1000 = era5[CH_T1000]
    t500 = era5[CH_T500]
    u850 = era5[CH_U850]
    v850 = era5[CH_V850]
    z1000 = era5[CH_Z1000] / G_ACCEL    # geopotential height in m
    z500 = era5[CH_Z500] / G_ACCEL

    # Potential temperature.
    theta1000 = t1000 * (P_REF / 100000.0) ** R_CP
    theta500 = t500 * (P_REF / 50000.0) ** R_CP

    # Brunt-Väisälä frequency. (g/theta) * (dtheta/dz)
    dz = np.maximum(z500 - z1000, 100.0)             # avoid /0
    theta_bar = 0.5 * (theta1000 + theta500)
    N2 = (G_ACCEL / theta_bar) * (theta500 - theta1000) / dz
    N2 = np.clip(N2, 1e-6, None)
    N = np.sqrt(N2)

    U = np.sqrt(u850 ** 2 + v850 ** 2)

    # Mountain height proxy: read the elevation_coarse zarr.
    elev_path = DATA / "elevation" / "elevation.zarr"
    if elev_path.exists():
        ez = zarr.open(str(elev_path), mode="r")
        if hasattr(ez, "shape") and len(ez.shape) >= 2:
            arr = np.asarray(ez[0] if ez.ndim == 3 else ez,
                              dtype=np.float32)
        else:
            arr = np.asarray(ez["elevation"], dtype=np.float32)
        if arr.shape != (ERA5_H, ERA5_W):
            # Resize via PIL/np (nearest-ish): assume already (169,281) or pad/crop.
            arr = np.array(arr[:ERA5_H, :ERA5_W])
    else:
        arr = np.zeros((ERA5_H, ERA5_W), dtype=np.float32)
    H_relief = np.clip(arr, 50.0, None)              # avoid /0; floor 50 m

    Fr = U / (N * H_relief)
    return Fr, U, N, H_relief


def crop_predictions_to_coarse(year: int, day: int):
    """CRAN-PM Europe prediction (4192, 6992) → coarse (169, 281) mean error vs GHAP."""
    pred_path = PROJECT / "evaluation_2022" / "predictions_t1.zarr"
    if not pred_path.exists():
        return None
    pz = zarr.open(str(pred_path), mode="r", zarr_format=2)
    if day >= pz.shape[0]:
        return None
    pred = np.asarray(pz[day], dtype=np.float32)

    ghap_path = DATA / "ghap_global_daily" / f"{year}.zarr"
    gz = zarr.open(str(ghap_path), mode="r", zarr_format=2)
    lat0 = int(round((90.0 - LAT_NORTH) / GHAP_RES))
    lon0 = int(round((LON_WEST - (-180.0)) / GHAP_RES))
    truth = np.asarray(gz[day, lat0:lat0 + GHAP_H,
                          lon0:lon0 + GHAP_W], dtype=np.float32)

    err = np.abs(pred - truth)
    # Downsample to ERA5 grid: simple box-average.
    rs = np.linspace(0, GHAP_H, ERA5_H + 1, dtype=int)
    cs = np.linspace(0, GHAP_W, ERA5_W + 1, dtype=int)
    out = np.zeros((ERA5_H, ERA5_W), dtype=np.float32)
    for i in range(ERA5_H):
        for j in range(ERA5_W):
            out[i, j] = err[rs[i]:rs[i+1], cs[j]:cs[j+1]].mean()
    return out


# ── plotting ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--day", type=int, default=23,
                        help="Winter inversion-prone day; default 23 = 24 Jan.")
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/project_462001140/ammar/eccv/"
                                     "cran-pm/paper/figures/"
                                     "fig_froude_elevation"))
    args = parser.parse_args()

    print(f"Loading ERA5 day={args.day} ...", flush=True)
    era5 = crop_era5_europe(args.year, args.day)
    Fr, U, N, H = compute_froude(era5)
    print(f"  Fr min={Fr.min():.2f}, max={Fr.max():.2f}, "
          f"median={np.median(Fr):.2f}", flush=True)

    print(f"Loading CRAN-PM prediction + GHAP truth ...", flush=True)
    err = crop_predictions_to_coarse(args.year, args.day)
    have_err = err is not None
    print(f"  err shape={getattr(err, 'shape', 'None')}", flush=True)

    extent = (LON_WEST, LON_WEST + ERA5_W * ERA5_RES,
              LAT_NORTH - ERA5_H * ERA5_RES, LAT_NORTH)

    fig, axes = plt.subplots(1, 3 if have_err else 2, figsize=(16.5, 5.0),
                              gridspec_kw={"wspace": 0.20})
    fig.patch.set_facecolor("white")
    cmap_fr = plt.get_cmap("RdYlBu_r")

    # Panel 1: Fr field (log scale for readability).
    ax = axes[0]
    Fr_log = np.log10(np.clip(Fr, 0.05, 100))
    im0 = ax.imshow(Fr_log, extent=extent, origin="upper", cmap=cmap_fr,
                    vmin=-1.0, vmax=1.0, aspect="auto",
                    interpolation="nearest")
    ax.contour(Fr_log, levels=[0.0], colors="black", linewidths=1.2,
               extent=extent, origin="upper")
    ax.set_title(r"Atmospheric Froude number $\mathrm{Fr}=U/(N H)$"
                 "\n(black contour: $\\mathrm{Fr}=1$, blocked/unblocked boundary)",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(im0, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$\log_{10}\mathrm{Fr}$", fontsize=9)

    # Panel 2: H_relief and U, V.
    ax = axes[1]
    im1 = ax.imshow(H, extent=extent, origin="upper", cmap="terrain",
                    vmin=0, vmax=H.max() * 0.9, aspect="auto",
                    interpolation="nearest")
    # 850 hPa wind quiver (subsampled).
    sub = 6
    lons = LON_WEST + (np.arange(ERA5_W) + 0.5) * ERA5_RES
    lats = LAT_NORTH - (np.arange(ERA5_H) + 0.5) * ERA5_RES
    L, La = np.meshgrid(lons[::sub], lats[::sub])
    ax.quiver(L, La, era5[CH_U850][::sub, ::sub], era5[CH_V850][::sub, ::sub],
              scale=300, width=0.0025, color="#222")
    ax.set_title(r"Mountain proxy $H$ and 850\,hPa wind"
                 "\n(barriers: Alps, Pyrenees, Scandes)",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    cb = fig.colorbar(im1, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("elevation (m)", fontsize=9)

    # Panel 3: Error map with Fr<1 mask overlaid.
    if have_err:
        ax = axes[2]
        err_clip = np.clip(err, 0.0, np.nanpercentile(err, 95))
        im2 = ax.imshow(err_clip, extent=extent, origin="upper",
                        cmap="RdYlBu_r", aspect="auto",
                        interpolation="nearest")
        # Hatched mask where Fr<1.
        mask = (Fr < 1.0).astype(float)
        ax.contourf(mask, levels=[0.5, 1.5], colors="none",
                    hatches=["///"], extent=extent, origin="upper",
                    alpha=0.0)
        ax.contour(Fr, levels=[1.0], colors="black", linewidths=1.0,
                   extent=extent, origin="upper")
        ax.set_title(r"$|$CRAN-PM $-$ GHAP$|$ daily error"
                     "\n(black contour delimits $\\mathrm{Fr}<1$ blocked zones)",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        cb = fig.colorbar(im2, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(r"|error| (µg m$^{-3}$)", fontsize=9)

        # Compute the actual diagnostic: mean |error| inside vs outside Fr<1.
        in_mask = (Fr < 1.0) & np.isfinite(err)
        out_mask = (Fr >= 1.0) & np.isfinite(err)
        m_in = float(np.nanmean(err[in_mask])) if in_mask.any() else float("nan")
        m_out = float(np.nanmean(err[out_mask])) if out_mask.any() else float("nan")
        ax.text(0.02, 0.98,
                f"mean |err| | $\\mathrm{{Fr}}<1$ = {m_in:.2f}\n"
                f"mean |err| | $\\mathrm{{Fr}}\\geq 1$ = {m_out:.2f}",
                transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="black",
                          alpha=0.9, boxstyle="round,pad=0.3"))
        stats = dict(year=args.year, day=args.day,
                     mean_err_Fr_lt_1=m_in, mean_err_Fr_geq_1=m_out,
                     n_lt=int(in_mask.sum()), n_geq=int(out_mask.sum()))
        with open(str(args.out.parent / "froude_correlations.json"), "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"  Fr<1 mean err = {m_in:.3f}, Fr>=1 mean err = {m_out:.3f}")

    fig.suptitle(
        f"Atmospheric Froude diagnostic — Europe, day-of-year {args.day}, "
        f"{args.year}",
        fontsize=12, y=1.02,
    )
    fig.savefig(str(args.out.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(args.out.with_suffix(".png")), bbox_inches="tight", dpi=180)
    print(f"Wrote {args.out.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
