"""Figure: Robinson world map showing the CRAN-PM study domain (Europe)
plus EEA station coverage and OOD test regions.

Outputs paper/figures/fig_robinson_study_domain.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import (
    apply_paper_style,
    load_global_validation_2025,
    make_robinson_axis,
    save_figure,
)


def main() -> None:
    apply_paper_style()

    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=10.0))
    make_robinson_axis(ax)

    # Europe primary domain (training + validation).
    europe_box = mpatches.Rectangle(
        (-25, 30.08), 70, 41.92,
        linewidth=1.6, edgecolor="#1f77b4", facecolor="#1f77b4",
        alpha=0.18, transform=ccrs.PlateCarree(), zorder=3,
    )
    ax.add_patch(europe_box)
    ax.text(
        10, 75, "Europe (training + test)",
        transform=ccrs.PlateCarree(), ha="center", va="bottom",
        fontsize=9, color="#1f77b4", fontweight="bold", zorder=4,
    )

    # OOD test regions.
    ood_regions = [
        ("USA / Canada", -120, 30, 60, 30, "#ff7f0e"),
        ("India", 68, 8, 30, 28, "#2ca02c"),
        ("East Asia", 100, 18, 45, 35, "#d62728"),
    ]
    for name, lon0, lat0, dlon, dlat, color in ood_regions:
        rect = mpatches.Rectangle(
            (lon0, lat0), dlon, dlat,
            linewidth=1.4, edgecolor=color, facecolor=color,
            alpha=0.16, linestyle="--", transform=ccrs.PlateCarree(), zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            lon0 + dlon / 2, lat0 + dlat + 1, name,
            transform=ccrs.PlateCarree(), ha="center", va="bottom",
            fontsize=8.5, color=color, zorder=4,
        )

    # Overlay global station density from the 2025 master validation.
    try:
        d = load_global_validation_2025()
        station_idx = np.unique(d["station_ids"], return_index=True)[1]
        lats = d["lats"][station_idx]
        lons = d["lons"][station_idx]
        ax.scatter(
            lons, lats,
            s=2.0, c="#222222", alpha=0.55,
            transform=ccrs.PlateCarree(), zorder=5,
            label=f"Air-quality stations (n = {len(station_idx):,})",
        )
        ax.legend(loc="lower left", fontsize=8.5,
                  frameon=True, facecolor="white", edgecolor="#888")
    except FileNotFoundError:
        pass

    fig.suptitle(
        "CRAN-PM study domain and out-of-distribution test regions",
        y=0.95, fontsize=11,
    )
    save_figure(fig, "fig_robinson_study_domain")
    plt.close(fig)
    print("Saved: fig_robinson_study_domain.{pdf,png}")


if __name__ == "__main__":
    main()
