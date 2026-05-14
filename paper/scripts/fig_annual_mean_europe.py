"""Figure: annual-mean PM2.5 over Europe — observed vs CRAN-PM vs bias.

ECCV-style: PlateCarree projection, RdYlBu_r colormap, panel labels (a)(b)(c),
white stat overlay box for the forecast panel.

Outputs paper/figures/fig_annual_mean_europe.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import (
    CMAP_BIAS,
    CMAP_PM,
    add_colorbar,
    add_stat_box,
    apply_paper_style,
    downsample,
    ghap_lat_lon,
    load_annual_means,
    make_europe_axis,
    panel_label,
    save_figure,
)


def main() -> None:
    apply_paper_style()

    gt, pred, count = load_annual_means()
    valid = count > 0
    gt = np.where(valid, gt, np.nan)
    pred = np.where(valid, pred, np.nan)

    factor = 4
    gt_lo = downsample(np.where(np.isnan(gt), 0, gt), factor)
    pred_lo = downsample(np.where(np.isnan(pred), 0, pred), factor)
    valid_lo = downsample(valid.astype(np.float32), factor) > 0.1
    gt_lo = np.where(valid_lo, gt_lo, np.nan)
    pred_lo = np.where(valid_lo, pred_lo, np.nan)

    lats, lons = ghap_lat_lon()
    lats_lo = lats[: lats.size // factor * factor].reshape(-1, factor).mean(axis=1)
    lons_lo = lons[: lons.size // factor * factor].reshape(-1, factor).mean(axis=1)
    lon2d, lat2d = np.meshgrid(lons_lo, lats_lo)

    bias = pred_lo - gt_lo

    # Annual-mean stats over valid pixels.
    diff_valid = (pred_lo - gt_lo)[valid_lo]
    gt_valid = gt_lo[valid_lo]
    rmse = float(np.sqrt(np.mean(diff_valid ** 2)))
    mae = float(np.mean(np.abs(diff_valid)))
    r = float(np.corrcoef(pred_lo[valid_lo], gt_valid)[0, 1])

    fig = plt.figure(figsize=(13.5, 5.2))
    proj = ccrs.PlateCarree()
    panels = [
        ("(a) Observed (GHAP), 2022", gt_lo, CMAP_PM, (0, 30),
         "Annual-mean PM$_{2.5}$ (µg m$^{-3}$)", None),
        ("(b) CRAN-PM T+1 forecast, 2022", pred_lo, CMAP_PM, (0, 30),
         "Annual-mean PM$_{2.5}$ (µg m$^{-3}$)",
         [f"RMSE = {rmse:.2f} µg m$^{{-3}}$",
          f"MAE  = {mae:.2f} µg m$^{{-3}}$",
          f"R    = {r:.3f}"]),
        ("(c) Bias (forecast − observed)", bias, CMAP_BIAS, (-8, 8),
         "Bias (µg m$^{-3}$)", None),
    ]

    for i, (title, data, cmap, vlim, cb_label, stats) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection=proj)
        make_europe_axis(ax, projection="pc")
        mesh = ax.pcolormesh(
            lon2d, lat2d, data,
            cmap=cmap, vmin=vlim[0], vmax=vlim[1],
            transform=ccrs.PlateCarree(), shading="auto", zorder=1,
        )
        panel_label(ax, title)
        add_colorbar(ax, mesh, cb_label, orientation="horizontal")
        if stats:
            add_stat_box(ax, stats, loc="lower left")

    fig.suptitle("Annual-mean PM$_{2.5}$ over Europe at 1 km (2022 test set)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_annual_mean_europe")
    plt.close(fig)
    print("Saved: fig_annual_mean_europe.{pdf,png}")


if __name__ == "__main__":
    main()
