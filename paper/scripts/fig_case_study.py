"""Generate the per-case-study paper figures.

Called by `sbatch_case_study.sh` after the inference job has produced the
predictions zarr for the case-study time window. Produces a 4-panel
spatial map (max-PM2.5 day in the window: GT / CRAN-PM / CAMS / persistence)
plus a multi-panel time series at 6 representative stations.

Outputs paper/figures_cases/<case>/case_<case>_<date>.{pdf,png}.

Inputs expected:
    --predictions   path to a zarr with vars `cranpm_t1`, `cams_t1`,
                    `persistence_t1`, `gt_t1`, `time` (lat, lon optional).
    --case          one of saharan-dust | iberian-fires | polish-winter.
    --date-start    inclusive ISO date.
    --date-end      inclusive ISO date.
    --output-dir    where to drop the figures (created if missing).

If the predictions zarr does not exist yet, the script writes a stub
indicating the SLURM run is pending.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import (
    CMAP_BIAS,
    CMAP_PM,
    METHOD_COLORS,
    add_colorbar,
    apply_paper_style,
    downsample,
    ghap_lat_lon,
    make_europe_axis,
    save_figure,
)

CASE_REGIONS = {
    "saharan-dust":   {"lon": (-12.0, 25.0), "lat": (30.0, 55.0), "vmax": 80,
                       "title": "Saharan dust intrusion (March 2022)",
                       "stations": ["ES1217A", "FR04143", "IT0457A", "PT01030", "GB0682A", "DE_BERLIN"]},
    "iberian-fires":  {"lon": (-12.0, 5.0), "lat": (35.0, 45.0), "vmax": 120,
                       "title": "Iberian Peninsula wildfires (July 2022)",
                       "stations": ["ES1438A", "ES1216A", "PT01040", "ES0009R", "ES0008R", "ES1779A"]},
    "polish-winter":  {"lon": (12.0, 25.0), "lat": (45.0, 55.0), "vmax": 100,
                       "title": "Central-European winter heating (January 2022)",
                       "stations": ["PL_KRAKOW", "PL_KATOWICE", "CZ_OSTRAVA", "SK_BRATISLAVA", "DE_DRESDEN", "PL_WARSAW"]},
}


def load_case(predictions: Path):
    import zarr
    z = zarr.open(str(predictions), mode="r")
    keys = list(z.array_keys()) if hasattr(z, "array_keys") else []
    return z, keys


def write_pending_stub(case: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stub = output_dir / f"PENDING_{case}.txt"
    stub.write_text(
        f"Case study '{case}' is awaiting SLURM job completion.\n"
        f"Submit: sbatch sbatch_case_study.sh {case} <date-start> <date-end>\n"
    )
    print(f"Wrote pending stub at {stub}")


def panel_map(ax, data, lats, lons, title, vmax, cmap=CMAP_PM, cbar_label="PM$_{2.5}$ (µg m$^{-3}$)"):
    lon2d, lat2d = np.meshgrid(lons, lats)
    mesh = ax.pcolormesh(
        lon2d, lat2d, data,
        cmap=cmap, vmin=0, vmax=vmax,
        transform=ccrs.PlateCarree(), shading="auto", zorder=1,
    )
    ax.set_title(title, pad=4)
    add_colorbar(ax, mesh, cbar_label)
    return mesh


def make_case_figure(case: str, predictions: Path, output_dir: Path) -> None:
    apply_paper_style()
    region = CASE_REGIONS[case]

    z, keys = load_case(predictions)
    needed = ["gt_t1", "cranpm_t1", "cams_t1", "persistence_t1", "time"]
    if not all(k in keys for k in needed):
        print(f"WARNING: predictions zarr missing keys; have {keys}")
        write_pending_stub(case, output_dir)
        return

    times = np.asarray(z["time"])
    daily_mean = np.array([float(np.nanmean(z["gt_t1"][i])) for i in range(len(times))])
    peak_idx = int(np.argmax(daily_mean))
    peak_date = str(times[peak_idx])

    gt = np.asarray(z["gt_t1"][peak_idx])
    pred = np.asarray(z["cranpm_t1"][peak_idx])
    cams = np.asarray(z["cams_t1"][peak_idx])
    persist = np.asarray(z["persistence_t1"][peak_idx])

    factor = 4
    lats_ghap, lons_ghap = ghap_lat_lon()
    lats_lo = lats_ghap[: lats_ghap.size // factor * factor].reshape(-1, factor).mean(axis=1)
    lons_lo = lons_ghap[: lons_ghap.size // factor * factor].reshape(-1, factor).mean(axis=1)

    fig = plt.figure(figsize=(13, 9))
    proj = ccrs.LambertAzimuthalEqualArea(central_longitude=10.0, central_latitude=52.0)
    titles = ["Observed (GHAP)", "CRAN-PM T+1", "CAMS forecast", "Persistence"]
    arrays = [gt, pred, cams, persist]
    for i, (title, arr) in enumerate(zip(titles, arrays), start=1):
        ax = fig.add_subplot(2, 2, i, projection=proj)
        make_europe_axis(ax, projection="laea")
        # Constrain to case-specific bbox
        ax.set_extent([region["lon"][0], region["lon"][1],
                       region["lat"][0], region["lat"][1]],
                      crs=ccrs.PlateCarree())
        arr_lo = downsample(arr, factor)
        panel_map(ax, arr_lo, lats_lo, lons_lo, f"{title}, {peak_date}", region["vmax"])

    fig.suptitle(region["title"], fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, f"fig_case_{case.replace('-', '_')}_maps")
    plt.close(fig)
    print(f"Saved case-study spatial figure for {case} (peak day {peak_date})")

    # ---- Time series ----
    fig2, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()
    for ax, sid in zip(axes, region["stations"]):
        if sid not in z.attrs.get("station_ids", []):
            ax.text(0.5, 0.5, f"Station {sid}\n(not in zarr)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # If station-level extraction is wired in the inference output:
        s = list(z.attrs.get("station_ids", [])).index(sid)
        ax.plot(times, z["station_gt"][:, s], "k-", label="GHAP", linewidth=1.2)
        ax.plot(times, z["station_cranpm"][:, s], color=METHOD_COLORS["CRAN-PM"],
                label="CRAN-PM", linewidth=1.2)
        ax.plot(times, z["station_cams"][:, s], color=METHOD_COLORS["CAMS"],
                label="CAMS", linewidth=1.0, linestyle="--")
        ax.set_title(sid, fontsize=9)
        ax.set_ylabel("PM$_{2.5}$ µg m$^{-3}$", fontsize=8)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        ax.grid(linestyle="--", alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=7)
    fig2.suptitle(f"{region['title']} — station time series", y=0.99)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig2, f"fig_case_{case.replace('-', '_')}_stations")
    plt.close(fig2)
    print(f"Saved station time-series figure for {case}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Case-study figure generator")
    parser.add_argument("--case", required=True, choices=list(CASE_REGIONS))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.predictions.exists():
        print(f"Predictions zarr not found: {args.predictions}")
        write_pending_stub(args.case, args.output_dir)
        return 1
    make_case_figure(args.case, args.predictions, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
