#!/usr/bin/env python3
"""Pre-render daily Europe PM2.5 maps (GHAP truth + CRAN-PM prediction)
for the HuggingFace Space `Date × country picker` tab.

Output: PNG renders in /tmp/cranpm_hf_space/europe_daily/<country>/<YYYY-MM-DD>.png

Covered countries (european capitals + reference cities, 8 boxes):
  France (Paris), UK (London), Germany (Berlin),
  Italy (Po Valley / Milano), Spain (Madrid), Poland (Warsaw),
  Netherlands (Amsterdam), Scandinavia (Helsinki).

Covered dates: 2022, one per month (12 dates) + the 3 case-study dates.

Each render is a 2-panel figure: GHAP truth | CRAN-PM T+1, with a metric
card in the corner.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import zarr

LAT_NORTH = 72.0
LON_WEST = -25.0
GHAP_RES = 0.01
GHAP_H, GHAP_W = 4192, 6992

DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")
PROJECT = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe")
OUT = Path("/tmp/cranpm_hf_space/europe_daily")
OUT.mkdir(parents=True, exist_ok=True)


# Monotonic white -> yellow -> orange -> red -> brown ramp so the sea
# (GHAP is 0 / NaN over water) renders pure white instead of dark blue.
GHAP_COLORS = [
    "#ffffff", "#fff7bc", "#fee391", "#fec44f", "#fe9929",
    "#ec7014", "#cc4c02", "#993404", "#662506",
]
CMAP = LinearSegmentedColormap.from_list("ghap", GHAP_COLORS, N=256)
CMAP = CMAP.copy()
CMAP.set_bad("white")  # masked sea / no-data -> white


def _mask_sea(gt, pr):
    """Mask ocean / no-data (GHAP <= 0 or non-finite) on both panels so
    they render white via CMAP.set_bad."""
    sea = ~np.isfinite(gt) | (gt <= 0.0)
    gt_m = np.ma.masked_where(sea, gt)
    pr_m = np.ma.masked_where(sea, np.clip(np.nan_to_num(pr), 0.0, None))
    return gt_m, pr_m, ~sea


# Country / city → (label, lat_min, lat_max, lon_min, lon_max).
COUNTRIES = {
    "France (Paris basin)":            (47.5, 50.5,  0.0,  4.5),
    "UK (London)":                     (50.5, 53.5, -2.5,  1.5),
    "Germany (Berlin / Ruhr)":         (50.0, 53.5,  6.0, 14.0),
    "Italy (Po Valley)":               (44.3, 46.2,  7.4, 12.6),
    "Spain (Madrid)":                  (38.5, 42.0, -5.5,  0.0),
    "Poland (Krakow / Silesia)":       (49.5, 52.5, 17.0, 22.5),
    "Netherlands (Randstad)":          (51.5, 53.5,  3.5,  7.0),
    "Finland (Helsinki / Baltic)":     (59.0, 62.0, 22.0, 28.5),
}

# Dates: 15th of each month + 3 case-study peak days.
DATES = [(m, 15) for m in range(1, 13)] + [(3, 15), (7, 18), (1, 12)]
DOY_OF = lambda mo, dy: __import__("datetime").date(2022, mo, dy).timetuple().tm_yday - 1


def latlon_to_rowcol(lat, lon):
    return (int(round((LAT_NORTH - lat) / GHAP_RES)),
            int(round((lon - LON_WEST) / GHAP_RES)))


def render_day(country_name, box, day_doy, ghap_z, pred_z, year=2022):
    lat_min, lat_max, lon_min, lon_max = box
    r0, c0 = latlon_to_rowcol(lat_max, lon_min)
    r1, c1 = latlon_to_rowcol(lat_min, lon_max)
    # Day-of-year alignment: pred[d] forecasts day d+1.
    gt = np.asarray(ghap_z[day_doy + 1,
                            (LAT_NORTH - lat_max) * 0 + r0 + 0,
                            (lon_min - LON_WEST) * 0 + c0 + 0],
                     dtype=np.float32)
    # Use the global GHAP rowcol from above (function-style).
    gt = np.asarray(ghap_z[day_doy + 1, r0:r1, c0:c1],
                     dtype=np.float32)
    # Pred zarr is on Europe grid (4192×6992) — same lat0/lon0 as the
    # Europe origin LAT_NORTH/LON_WEST.
    pr = np.asarray(pred_z[day_doy, r0:r1, c0:c1], dtype=np.float32)
    pr = np.nan_to_num(np.clip(pr, 0.0, None), nan=0.0)

    finite = np.isfinite(gt) & (gt > 0.5)
    if finite.sum() == 0:
        return None
    rmse = float(np.sqrt(np.mean((pr[finite] - gt[finite]) ** 2)))
    mae = float(np.mean(np.abs(pr[finite] - gt[finite])))
    mean_gt = float(np.mean(gt[finite]))

    vmax = max(40.0, float(np.nanpercentile(gt, 98)))
    norm = Normalize(vmin=0, vmax=vmax)
    extent = (lon_min, lon_max, lat_min, lat_max)

    gt_m, pr_m, _ = _mask_sea(gt, pr)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0),
                              gridspec_kw={"wspace": 0.18})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
    axes[0].imshow(gt_m[::2, ::2], extent=extent, origin="upper",
                   cmap=CMAP, norm=norm, aspect="auto",
                   interpolation="nearest")
    axes[0].set_title("GHAP truth", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Longitude (°E)", fontsize=9)
    axes[0].set_ylabel("Latitude (°N)", fontsize=9)
    axes[0].tick_params(labelsize=8)

    im = axes[1].imshow(pr_m[::2, ::2], extent=extent, origin="upper",
                        cmap=CMAP, norm=norm, aspect="auto",
                        interpolation="nearest")
    axes[1].set_title("CRAN-PM T+1", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Longitude (°E)", fontsize=9)
    axes[1].tick_params(labelsize=8)
    axes[1].text(0.97, 0.04,
                  f"RMSE = {rmse:.1f} µg m$^{{-3}}$\n"
                  f"MAE  = {mae:.1f} µg m$^{{-3}}$\n"
                  f"⟨GHAP⟩ = {mean_gt:.1f} µg m$^{{-3}}$",
                  transform=axes[1].transAxes, ha="right", va="bottom",
                  fontsize=8.5,
                  bbox=dict(facecolor="white", edgecolor="black",
                            alpha=0.92, boxstyle="round,pad=0.3"))

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label(r"PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return fig


def main():
    ghap_z = zarr.open(str(DATA / "ghap_global_daily" / "2022.zarr"),
                        mode="r", zarr_format=2)
    pred_z = zarr.open(str(PROJECT / "evaluation_2022" / "predictions_t1.zarr"),
                        mode="r", zarr_format=2)

    # Note: pred_z is Europe-cropped already (4192×6992), starting at
    # LAT_NORTH / LON_WEST. We compute rows/cols relative to those.
    # But ghap_z is GLOBAL — its origin is 90°N/-180°E. We need to
    # offset back to those globals.
    # For simplicity I'll wrap by re-reading GHAP via its Europe origin.
    eu_r0 = int(round((90.0 - LAT_NORTH) / GHAP_RES))
    eu_c0 = int(round((LON_WEST - (-180.0)) / GHAP_RES))
    # Create a small wrapper that returns a Europe-relative crop
    class GhapEU:
        def __init__(self, z, r0, c0):
            self.z = z; self.r0 = r0; self.c0 = c0
        def __getitem__(self, key):
            d, rs, cs = key
            return self.z[d, self.r0 + rs.start: self.r0 + rs.stop,
                           self.c0 + cs.start: self.c0 + cs.stop]
        @property
        def shape(self):
            return self.z.shape
    # Replace direct ghap_z access in render_day with a helper:
    # (the simpler thing is to inline-pass the Europe-relative crop).

    seen_dates = set()
    for (mo, dy) in DATES:
        if (mo, dy) in seen_dates:
            continue
        seen_dates.add((mo, dy))
        doy = DOY_OF(mo, dy)
        date_str = f"2022-{mo:02d}-{dy:02d}"
        for cname, box in COUNTRIES.items():
            slug = (cname.split("(")[0].strip().lower()
                     .replace(" ", "_").replace("/", "_"))
            out_dir = OUT / slug
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{date_str}.png"
            if out_path.exists():
                continue
            # GHAP rowcol uses GLOBAL origin; pred uses EUROPE origin.
            lat_min, lat_max, lon_min, lon_max = box
            gr0 = int(round((90.0 - lat_max) / GHAP_RES))
            gr1 = int(round((90.0 - lat_min) / GHAP_RES))
            gc0 = int(round((lon_min - (-180.0)) / GHAP_RES))
            gc1 = int(round((lon_max - (-180.0)) / GHAP_RES))
            er0 = gr0 - eu_r0; er1 = gr1 - eu_r0
            ec0 = gc0 - eu_c0; ec1 = gc1 - eu_c0
            gt = np.asarray(ghap_z[doy + 1, gr0:gr1, gc0:gc1],
                             dtype=np.float32)
            pr = np.nan_to_num(np.asarray(pred_z[doy, er0:er1, ec0:ec1],
                                            dtype=np.float32), nan=0.0)
            pr = np.clip(pr, 0.0, None)
            finite = np.isfinite(gt) & (gt > 0.5)
            if finite.sum() == 0:
                continue
            rmse = float(np.sqrt(np.mean((pr[finite] - gt[finite]) ** 2)))
            mae = float(np.mean(np.abs(pr[finite] - gt[finite])))
            mean_gt = float(np.mean(gt[finite]))
            vmax = max(40.0, float(np.nanpercentile(gt, 98)))
            norm = Normalize(vmin=0, vmax=vmax)
            extent = (lon_min, lon_max, lat_min, lat_max)
            gt_m, pr_m, _ = _mask_sea(gt, pr)
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0),
                                      gridspec_kw={"wspace": 0.18})
            fig.patch.set_facecolor("white")
            for ax in axes:
                ax.set_facecolor("white")
            axes[0].imshow(gt_m[::2, ::2], extent=extent, origin="upper",
                            cmap=CMAP, norm=norm, aspect="auto",
                            interpolation="nearest")
            axes[0].set_title(f"GHAP truth — {cname}",
                               fontsize=10.5, fontweight="bold")
            axes[0].set_xlabel("Longitude (°E)", fontsize=9)
            axes[0].set_ylabel("Latitude (°N)", fontsize=9)
            axes[0].tick_params(labelsize=8)
            im = axes[1].imshow(pr_m[::2, ::2], extent=extent, origin="upper",
                                 cmap=CMAP, norm=norm, aspect="auto",
                                 interpolation="nearest")
            axes[1].set_title(f"CRAN-PM T+1 — {date_str}",
                               fontsize=10.5, fontweight="bold")
            axes[1].set_xlabel("Longitude (°E)", fontsize=9)
            axes[1].tick_params(labelsize=8)
            axes[1].text(0.97, 0.04,
                          f"RMSE = {rmse:.1f} µg m$^{{-3}}$\n"
                          f"MAE  = {mae:.1f} µg m$^{{-3}}$",
                          transform=axes[1].transAxes, ha="right", va="bottom",
                          fontsize=8.5,
                          bbox=dict(facecolor="white", edgecolor="black",
                                    alpha=0.92, boxstyle="round,pad=0.3"))
            cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
            cb.set_label(r"PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9)
            cb.ax.tick_params(labelsize=8)
            fig.savefig(str(out_path), bbox_inches="tight", dpi=110)
            plt.close(fig)
            print(f"  {slug}/{date_str}.png", flush=True)

    print(f"\nTotal: {sum(len(list(d.iterdir())) for d in OUT.iterdir() if d.is_dir())} PNGs in {OUT}")


if __name__ == "__main__":
    main()
