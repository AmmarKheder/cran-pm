"""Figure: 4-panel input data flow for one representative day.

Shows what CRAN-PM ingests on a single day to forecast t+1:
  (a) ERA5 wind field (u10/v10 arrows) over a temperature background
  (b) CAMS analysis PM2.5 at 0.4 deg (background chemistry)
  (c) GHAP PM2.5 today at 1 km (local high-res input + t-1 target proxy)
  (d) GMTED2010 elevation downsampled to the GHAP grid

Outputs paper/figures/fig_input_data_flow.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import (
    CMAP_PM,
    add_colorbar,
    apply_paper_style,
    downsample,
    ghap_lat_lon,
    make_europe_axis,
    panel_label,
    save_figure,
)


DAY_OF_YEAR_2022 = 76   # 17 March 2022 — peak of the Saharan dust intrusion.
YEAR = 2022


def main() -> None:
    apply_paper_style()
    from cranpm.inference.europe_inputs import EuropeInputsBuilder

    builder = EuropeInputsBuilder()
    inp = builder.build(year=YEAR, day_idx=DAY_OF_YEAR_2022)

    # Sub-extract the bits we want to show.
    # ERA5 layout: channels [0..29] for day t.
    # Channel order in the global ERA5 zarr typically: u10, v10, t2m, sp, msl, ...
    # We use channels 0/1 for wind, channel 2 for temperature background.
    u10 = inp.era5_global[0]
    v10 = inp.era5_global[1]
    t2m = inp.era5_global[2]

    # CAMS PM2.5 today: channels 30 (first 5 are CAMS in our layout).
    cams_pm25 = inp.era5_global[30]

    # Downsample GHAP + elev for fast rendering at the Europe extent.
    factor = 4
    ghap_lo = downsample(inp.ghap_t0, factor)
    elev_lo = downsample(inp.elev_hires, factor)
    lats, lons = ghap_lat_lon()
    lats_lo = lats[: lats.size // factor * factor].reshape(-1, factor).mean(axis=1)
    lons_lo = lons[: lons.size // factor * factor].reshape(-1, factor).mean(axis=1)

    # ERA5 grid is 0.25 deg, 168x280.
    e5_lats = 72.0 - 0.25 * (np.arange(168) + 0.5)
    e5_lons = -25.0 + 0.25 * (np.arange(280) + 0.5)

    fig = plt.figure(figsize=(13.5, 9.5))
    proj = ccrs.PlateCarree()

    # ---- (a) ERA5 wind on T2m background.
    ax = fig.add_subplot(2, 2, 1, projection=proj)
    make_europe_axis(ax, projection="pc")
    LE, LO = np.meshgrid(e5_lons, e5_lats)
    mesh = ax.pcolormesh(LE, LO, t2m, cmap="RdYlBu_r",
                          transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    step = 6
    ax.quiver(LE[::step, ::step], LO[::step, ::step],
               u10[::step, ::step], v10[::step, ::step],
               scale=200, width=0.0028, color="black",
               transform=ccrs.PlateCarree(), zorder=4)
    add_colorbar(ax, mesh, "ERA5 T$_{2m}$ (normalised)")
    panel_label(ax, "(a) ERA5 wind on T$_{2m}$, 17 Mar 2022")

    # ---- (b) CAMS PM2.5 analysis.
    ax = fig.add_subplot(2, 2, 2, projection=proj)
    make_europe_axis(ax, projection="pc")
    LE, LO = np.meshgrid(e5_lons, e5_lats)
    mesh = ax.pcolormesh(LE, LO, cams_pm25, cmap=CMAP_PM,
                          transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    add_colorbar(ax, mesh, "CAMS PM$_{2.5}$ (normalised)")
    panel_label(ax, "(b) CAMS PM$_{2.5}$ analysis (25 km input)")

    # ---- (c) GHAP at t.
    ax = fig.add_subplot(2, 2, 3, projection=proj)
    make_europe_axis(ax, projection="pc")
    LE, LO = np.meshgrid(lons_lo, lats_lo)
    mesh = ax.pcolormesh(LE, LO, ghap_lo, cmap=CMAP_PM, vmin=0, vmax=80,
                          transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    add_colorbar(ax, mesh, "GHAP PM$_{2.5}$ (µg m$^{-3}$)")
    panel_label(ax, "(c) GHAP PM$_{2.5}$ at 1 km (local-branch input)")

    # ---- (d) Elevation.
    ax = fig.add_subplot(2, 2, 4, projection=proj)
    make_europe_axis(ax, projection="pc")
    LE, LO = np.meshgrid(lons_lo, lats_lo)
    mesh = ax.pcolormesh(LE, LO, elev_lo, cmap="terrain",
                          vmin=-200, vmax=3500,
                          transform=ccrs.PlateCarree(), shading="auto", zorder=1)
    add_colorbar(ax, mesh, "GMTED2010 elevation (m)")
    panel_label(ax, "(d) High-resolution elevation (250 m)")

    fig.suptitle("CRAN-PM inputs on a single day (17 March 2022, Saharan-dust onset)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig_input_data_flow")
    plt.close(fig)
    print("Saved: fig_input_data_flow.{pdf,png}")


if __name__ == "__main__":
    main()
