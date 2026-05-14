"""Figure: EEA station coverage in Europe (PlateCarree, ECCV-style).

Reads Master_Validation_2025.npz, filters to the Europe bounding box,
plots each unique station coloured by its mean PM2.5 observation,
with the RdYlBu_r colormap.

Outputs paper/figures/fig_eea_stations_europe.{pdf,png}.
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
    apply_paper_style,
    load_global_validation_2025,
    make_europe_axis,
    panel_label,
    save_figure,
)


EUROPE_BOX = (-25.0, 45.0, 30.0, 72.0)  # (lon_w, lon_e, lat_s, lat_n)


def main() -> None:
    apply_paper_style()
    d = load_global_validation_2025()
    lats = d["lats"]
    lons = d["lons"]
    pm = d["pm25_observed"]
    sid = d["station_ids"]

    mask = (
        (lons >= EUROPE_BOX[0]) & (lons <= EUROPE_BOX[1])
        & (lats >= EUROPE_BOX[2]) & (lats <= EUROPE_BOX[3])
        & np.isfinite(pm)
    )
    lats, lons, pm, sid = lats[mask], lons[mask], pm[mask], sid[mask]

    # Aggregate per station: take the mean PM2.5.
    uniq, inv = np.unique(sid, return_inverse=True)
    station_pm = np.bincount(inv, weights=pm) / np.bincount(inv)
    station_lat = np.bincount(inv, weights=lats) / np.bincount(inv)
    station_lon = np.bincount(inv, weights=lons) / np.bincount(inv)

    fig = plt.figure(figsize=(11, 7.4))
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    make_europe_axis(ax, projection="pc")

    sc = ax.scatter(
        station_lon, station_lat, c=station_pm,
        s=14, cmap=CMAP_PM, vmin=0, vmax=30,
        edgecolor="#222", linewidths=0.25,
        transform=ccrs.PlateCarree(), zorder=4,
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                         shrink=0.7, pad=0.02, aspect=30, extend="both")
    cbar.set_label("Mean observed PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.4)

    panel_label(ax, f"European air-quality stations (n = {len(uniq):,})")
    fig.suptitle("EEA monitoring network used for validation (mean PM$_{2.5}$ in 2025)",
                 fontsize=11, y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_eea_stations_europe")
    plt.close(fig)
    print(f"Saved: fig_eea_stations_europe.{{pdf,png}} (n_stations = {len(uniq)})")


if __name__ == "__main__":
    main()
