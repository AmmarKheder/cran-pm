"""Figure: best forecast day 2022 — observed vs CRAN-PM vs bias.

ECCV-style with RdYlBu_r and panel labels.

Outputs paper/figures/fig_best_day_comparison.{pdf,png}.
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
    load_best_day,
    make_europe_axis,
    panel_label,
    save_figure,
)


def main() -> None:
    apply_paper_style()
    d = load_best_day()
    date = d["date"]

    factor = 4
    gt = downsample(d["gt_t1"], factor)
    pred = downsample(d["pred_t1"], factor)
    bias = pred - gt

    valid = downsample((d["gt_t1"] > 0).astype(np.float32), factor) > 0.1
    gt = np.where(valid, gt, np.nan)
    pred = np.where(valid, pred, np.nan)
    bias = np.where(valid, bias, np.nan)

    lats, lons = ghap_lat_lon()
    lats_lo = lats[: lats.size // factor * factor].reshape(-1, factor).mean(axis=1)
    lons_lo = lons[: lons.size // factor * factor].reshape(-1, factor).mean(axis=1)
    lon2d, lat2d = np.meshgrid(lons_lo, lats_lo)

    diff_valid = (pred - gt)[valid]
    gt_valid = gt[valid]
    rmse = float(np.sqrt(np.mean(diff_valid ** 2)))
    mae = float(np.mean(np.abs(diff_valid)))
    r = float(np.corrcoef(pred[valid], gt_valid)[0, 1])

    fig = plt.figure(figsize=(13.5, 5.2))
    proj = ccrs.PlateCarree()
    panels = [
        (f"(a) Observed (GHAP), {date}", gt, CMAP_PM, (0, 60),
         "PM$_{2.5}$ (µg m$^{-3}$)", None),
        (f"(b) CRAN-PM T+1 forecast, {date}", pred, CMAP_PM, (0, 60),
         "PM$_{2.5}$ (µg m$^{-3}$)",
         [f"RMSE = {rmse:.2f} µg m$^{{-3}}$",
          f"MAE  = {mae:.2f} µg m$^{{-3}}$",
          f"R    = {r:.3f}"]),
        ("(c) Forecast − observed", bias, CMAP_BIAS, (-15, 15),
         "Difference (µg m$^{-3}$)", None),
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
        add_colorbar(ax, mesh, cb_label)
        if stats:
            add_stat_box(ax, stats, loc="lower left")

    fig.suptitle(f"Best forecast day in 2022: {date}", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_best_day_comparison")
    plt.close(fig)
    print(f"Saved: fig_best_day_comparison.{{pdf,png}} (date={date})")


if __name__ == "__main__":
    main()
