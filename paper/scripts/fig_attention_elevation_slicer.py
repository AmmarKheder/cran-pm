#!/usr/bin/env python3
"""Visualise the elevation-aware attention bias used by CRAN-PM.

The TopoFlowBlock biases each attention logit by
   α_{ij} = (Q_i · K_j)/√d  −  λ · |z_i − z_j|        (Eq. 5 in the paper)
where z is the local terrain height (m) and λ > 0 is a learnable
elevation-bias coefficient. This penalises information flow across
strong topographic barriers and is meant to reproduce the boundary-layer
trapping mechanism (Sect. 8.4, Fig. fig:phys_ablation_heatmap).

This figure shows the SOFTMAXed attention from a fixed *source*
(query) tile in the Po Valley to every tile over Europe, comparing
three settings:

  (a) isotropic baseline                 :  α_{ij} = −‖x_i − x_j‖²/(2σ²)
  (b) CRAN-PM elevation-aware            :  α_{ij} = (a) − λ |z_i − z_j|
  (c) ConvLSTM-style purely-spatial bias :  α_{ij} = −‖x_i − x_j‖²/(2σ²)

The expected pattern is that (b) shows a sharp attention drop along the
Alpine and Pyrenean ridges, while (a) and (c) bleed through them. We
also overlay an East–West transect at 45.4°N (the Sichuan-style cut
across the Po Valley) showing attention vs longitude with elevation
underneath.

This is an analytical illustration of the bias *kernel* — not a
forward pass through the trained model — so the figure can be produced
on the login node in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import zarr


DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")

# Europe window — identical to training config.
LAT_NORTH = 72.0
LON_WEST = -25.0
ERA5_RES = 0.25
ERA5_H, ERA5_W = 169, 281

# Source = Po Valley centre (Milano).
SOURCE_LAT = 45.4
SOURCE_LON = 9.2

# Attention kernel parameters (learned values from v10f checkpoint, see
# topoflow_europe/models/cross_attention.py — comments report
# σ ~ 6 patches and λ ~ 0.0015 1/m as the final-epoch values).
SIGMA_PATCHES = 6.0
LAMBDA_ELEV = 0.0015          # 1/m


def load_europe_elevation():
    """Read the ERA5-resolution elevation zarr (169, 281)."""
    p = DATA / "elevation" / "elevation.zarr"
    z = zarr.open(str(p), mode="r")
    if hasattr(z, "shape"):
        arr = np.asarray(z[0] if z.ndim == 3 else z, dtype=np.float32)
    else:
        arr = np.asarray(z["elevation"], dtype=np.float32)
    return arr[:ERA5_H, :ERA5_W]


def attention_maps(elev, source_rc, sigma_pix=SIGMA_PATCHES,
                    lam=LAMBDA_ELEV):
    sr, sc = source_rc
    H, W = elev.shape
    rows = np.arange(H)
    cols = np.arange(W)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    dx2 = ((R - sr) ** 2 + (C - sc) ** 2).astype(np.float32)
    dz = np.abs(elev - elev[sr, sc])

    log_iso = -dx2 / (2.0 * sigma_pix ** 2)
    log_elev = log_iso - lam * dz
    # Softmax across all targets so the maps are comparable.
    def _softmax(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()
    return _softmax(log_iso), _softmax(log_elev), dz, dx2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/project_462001140/ammar/eccv/"
                                     "cran-pm/paper/figures/"
                                     "fig_attention_elevation_slicer"))
    args = parser.parse_args()

    elev = load_europe_elevation()
    sr = int(round((LAT_NORTH - SOURCE_LAT) / ERA5_RES))
    sc = int(round((SOURCE_LON - LON_WEST) / ERA5_RES))
    src_alt = float(elev[sr, sc])
    print(f"Source: lat={SOURCE_LAT}°N, lon={SOURCE_LON}°E, "
          f"row={sr}, col={sc}, elev={src_alt:.0f} m", flush=True)

    a_iso, a_elev, dz, _ = attention_maps(elev, (sr, sc))
    extent = (LON_WEST, LON_WEST + ERA5_W * ERA5_RES,
              LAT_NORTH - ERA5_H * ERA5_RES, LAT_NORTH)

    fig = plt.figure(figsize=(15.5, 9.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1.0],
                            hspace=0.28, wspace=0.22)
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap("RdYlBu_r")
    # Plot in log-attention so contrast is visible across the full domain.
    norm = mcolors.LogNorm(vmin=1e-7, vmax=1e-2)

    panels = [
        (a_iso,  "Isotropic distance kernel only\n"
                  r"$\alpha_{ij}\propto\exp(-\|x_i-x_j\|^2/2\sigma^2)$"),
        (a_elev, "CRAN-PM elevation-aware attention\n"
                  r"$\alpha_{ij}\propto\exp\left(-\frac{\|x_i-x_j\|^2}"
                  r"{2\sigma^2}-\lambda\,|z_i-z_j|\right)$"),
        (a_iso - a_elev,
                  "Difference (Isotropic $-$ CRAN-PM)\n"
                  "positive bands $\\Rightarrow$ where the elevation bias "
                  "cuts attention flow"),
    ]
    for k, (data, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        if k < 2:
            im = ax.imshow(data, extent=extent, origin="upper",
                            cmap=cmap, norm=norm, aspect="auto",
                            interpolation="nearest")
        else:
            v = np.nanpercentile(np.abs(data), 99) + 1e-12
            im = ax.imshow(data, extent=extent, origin="upper",
                            cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto",
                            interpolation="nearest")
        # Mark Po Valley source.
        ax.plot([SOURCE_LON], [SOURCE_LAT], marker="*",
                markersize=14, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.5, zorder=10)
        # Elevation contours at 500 / 1500 / 2500 m (Alpine ridge).
        lons = LON_WEST + (np.arange(ERA5_W) + 0.5) * ERA5_RES
        lats = LAT_NORTH - (np.arange(ERA5_H) + 0.5) * ERA5_RES
        LON, LAT = np.meshgrid(lons, lats)
        ax.contour(LON, LAT, elev, levels=[500, 1500, 2500],
                   colors=["#666", "#444", "#222"], linewidths=[0.4, 0.7, 1.0],
                   alpha=0.7)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9.5)
        if k == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        if k < 2:
            cb.set_label("attention weight  $\\alpha_{ij}$", fontsize=9)
        else:
            cb.set_label("$\\Delta\\alpha$", fontsize=9)
        ax.tick_params(labelsize=8)

    # Bottom: W-E transect at 45.4°N.
    ax = fig.add_subplot(gs[1, :])
    rr = int(round((LAT_NORTH - SOURCE_LAT) / ERA5_RES))
    lons = LON_WEST + (np.arange(ERA5_W) + 0.5) * ERA5_RES
    a_iso_row = a_iso[rr]
    a_elev_row = a_elev[rr]
    elev_row = elev[rr]

    ax.fill_between(lons, 0, elev_row, color="#bbb", alpha=0.6,
                     label="terrain (m)")
    ax2 = ax.twinx()
    ax2.semilogy(lons, np.clip(a_iso_row, 1e-12, None),
                  color="#1f77b4", lw=1.6, label="isotropic")
    ax2.semilogy(lons, np.clip(a_elev_row, 1e-12, None),
                  color="#d62728", lw=1.6,
                  label="CRAN-PM (elev bias)")
    ax2.set_ylabel("attention weight (log)", fontsize=9.5)
    ax.set_ylabel("elevation (m)", fontsize=9.5)
    ax.set_xlabel(r"Longitude (°E) — W$\rightarrow$E transect at 45.4°N",
                  fontsize=10)
    ax.axvline(SOURCE_LON, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_title(
        "Cross-section along 45.4°N (Po Valley source $\\bigstar$): "
        "CRAN-PM attention drops at each Alpine and Pyrenean ridge, "
        "isotropic attention bleeds across",
        fontsize=10.5,
    )
    lines = (ax.get_legend_handles_labels()[0] +
              ax2.get_legend_handles_labels()[0])
    labels = (ax.get_legend_handles_labels()[1] +
              ax2.get_legend_handles_labels()[1])
    ax.legend(lines, labels, fontsize=9, loc="upper right")

    fig.suptitle(
        "Elevation-aware attention bias of CRAN-PM "
        "(analytical illustration on Europe DEM)",
        fontsize=12.5, y=1.02,
    )
    fig.savefig(str(args.out.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(args.out.with_suffix(".png")), bbox_inches="tight",
                 dpi=180)
    print(f"Wrote {args.out.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
