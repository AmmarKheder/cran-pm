#!/usr/bin/env python3
"""Pinpoint single-tile CRAN-PM forecast for the HuggingFace Space.

The user picks a *centre latitude/longitude* and a forecast date. We crop
exactly one 512x512 GHAP tile (the native local-branch window, 0.01 deg
per pixel = a 5.12 deg box) around that point and render it at full
native resolution:

    GHAP truth  |  CRAN-PM T+1  |  + EEA ground-truth stations

Design choices
--------------
* One 512x512 tile only. The full pan-European grid is 4192x6992; saving
  that at a publication DPI overflows matplotlib's image backend (the
  "Image size of ... pixels is too large" error). A single tile keeps
  the rendered figure well within limits while still being "ultra
  resolution" -- every GHAP pixel is drawn 1:1, no decimation.
* Ocean is white. GHAP is NaN over sea; we mask it and set the colormap
  bad-colour + axes facecolor to white instead of the dark-blue low end.
* European EEA observations for the forecast day are overlaid as points
  using the same colour scale, so the user sees model vs ground truth.

Reads precomputed predictions (no 1.14 GB checkpoint load) so it runs
fast and deterministically on the HF Space CPU free tier.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ── Grid geometry ──────────────────────────────────────────────────────
GHAP_RES = 0.01
TILE = 512  # local-branch tile size in pixels (= 5.12 deg box)

# GHAP global zarr: origin 90 N / -180 E.
GHAP_GLOBAL_LAT0 = 90.0
GHAP_GLOBAL_LON0 = -180.0

# CRAN-PM Europe prediction zarr: origin 72 N / -25 E, 4192 x 6992.
EU_LAT0 = 72.0
EU_LON0 = -25.0
EU_H, EU_W = 4192, 6992

DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")
PROJECT = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe")
# EEA PM2.5 monitoring-station coordinates (the European ground-truth
# network used to validate CRAN-PM in 2022). One row per sampling point;
# we de-duplicate to one marker per station.
EEA_METADATA_CSV = "/scratch/project_462001140/ammar/eccv/eea_airbase/station_metadata.csv"

# Monotonic white -> yellow -> orange -> red -> dark brown ramp
# (YlOrBr-style, perceptually increasing). Clean air stays near-white so
# the masked white ocean blends into a single uncluttered background and
# only real pollution draws the eye.
_GHAP_COLORS = [
    "#ffffff", "#fff7bc", "#fee391", "#fec44f", "#fe9929",
    "#ec7014", "#cc4c02", "#993404", "#662506",
]
CMAP = LinearSegmentedColormap.from_list("ghap_white", _GHAP_COLORS, N=256)
CMAP = CMAP.copy()
CMAP.set_bad("white")  # NaN / ocean / no-data renders pure white


# ── Geometry helpers ───────────────────────────────────────────────────
def _doy0(date_str: str) -> int:
    """0-based day-of-year for a 'YYYY-MM-DD' string."""
    y, m, d = (int(x) for x in date_str.split("-"))
    return _dt.date(y, m, d).timetuple().tm_yday - 1


def tile_window(center_lat: float, center_lon: float) -> dict:
    """Top-left GHAP-global row/col of the 512 tile centred on the point,
    clamped so the tile stays inside the European prediction grid.

    Returns a dict with both the GHAP-global and the Europe-relative
    top-left indices plus the geographic extent of the tile.
    """
    # Europe-relative top-left, clamped to [0, EU-TILE].
    er0 = int(round((EU_LAT0 - center_lat) / GHAP_RES)) - TILE // 2
    ec0 = int(round((center_lon - EU_LON0) / GHAP_RES)) - TILE // 2
    er0 = int(np.clip(er0, 0, EU_H - TILE))
    ec0 = int(np.clip(ec0, 0, EU_W - TILE))

    # Same window expressed on the GHAP-global grid.
    gr0 = int(round((GHAP_GLOBAL_LAT0 - EU_LAT0) / GHAP_RES)) + er0
    gc0 = int(round((EU_LON0 - GHAP_GLOBAL_LON0) / GHAP_RES)) + ec0

    lat_max = EU_LAT0 - er0 * GHAP_RES
    lat_min = lat_max - TILE * GHAP_RES
    lon_min = EU_LON0 + ec0 * GHAP_RES
    lon_max = lon_min + TILE * GHAP_RES
    return {
        "er0": er0, "ec0": ec0, "gr0": gr0, "gc0": gc0,
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "extent": (lon_min, lon_max, lat_min, lat_max),
    }


_EEA_CACHE: dict | None = None


def _load_eea_pm25_stations():
    """All EEA PM2.5 station coordinates, one (lat, lon) per station.

    Cached. Reads the AirBase station metadata, keeps PM2.5 sampling
    points with valid coordinates and de-duplicates by station code.
    """
    global _EEA_CACHE
    if _EEA_CACHE is not None:
        return _EEA_CACHE
    import csv

    seen: dict[str, tuple[float, float]] = {}
    with open(EEA_METADATA_CSV, newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            poll = (row.get("Air Pollutant") or "").upper()
            if "PM2" not in poll:
                continue
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except (TypeError, ValueError):
                continue
            code = row.get("Air Quality Station EoI Code") or f"{lat},{lon}"
            seen.setdefault(code, (lat, lon))
    if seen:
        arr = np.array(list(seen.values()), dtype=np.float32)
        _EEA_CACHE = (arr[:, 0], arr[:, 1])
    else:
        _EEA_CACHE = (np.array([]), np.array([]))
    return _EEA_CACHE


def _stations_in_box(win: dict):
    """EEA PM2.5 station coordinates inside the tile box.

    Returns (lons, lats). These are the locations of the European
    ground-truth monitoring network; per-day observed concentrations
    for 2022 are not bundled with the Space (the validation archive is
    served separately), so we plot the network coverage as reference
    markers.
    """
    slat, slon = _load_eea_pm25_stations()
    if slat.size == 0:
        return np.array([]), np.array([])
    box = (
        (slon >= win["lon_min"]) & (slon <= win["lon_max"])
        & (slat >= win["lat_min"]) & (slat <= win["lat_max"])
    )
    return slon[box], slat[box]


# ── Main render ────────────────────────────────────────────────────────
def render_pinpoint(
    date_str: str,
    center_lat: float,
    center_lon: float,
    *,
    ghap_zarr: str | None = None,
    pred_zarr: str | None = None,
    show_stations: bool = True,
):
    """Render a single 512x512 tile: GHAP truth | CRAN-PM T+1 | stations.

    `date_str` is the *forecast valid date* (the day being predicted).
    Returns a matplotlib Figure.
    """
    valid0 = _doy0(date_str)  # 0-based doy of the valid day
    pred_idx = valid0 - 1  # pred[d] forecasts day d+1
    if not (1 <= valid0 <= 362):
        raise ValueError(
            f"{date_str}: forecast valid date must be 2022-01-02 .. 2022-12-29"
        )

    win = tile_window(center_lat, center_lon)

    gz = zarr.open(
        ghap_zarr or str(DATA / "ghap_global_daily" / "2022.zarr"),
        mode="r", zarr_format=2,
    )
    pz = zarr.open(
        pred_zarr or str(PROJECT / "evaluation_2022" / "predictions_t1.zarr"),
        mode="r", zarr_format=2,
    )

    gt = np.asarray(
        gz[valid0, win["gr0"]:win["gr0"] + TILE, win["gc0"]:win["gc0"] + TILE],
        dtype=np.float32,
    )
    pr = np.asarray(
        pz[pred_idx, win["er0"]:win["er0"] + TILE, win["ec0"]:win["ec0"] + TILE],
        dtype=np.float32,
    )

    # Ocean / no-data -> NaN -> white via CMAP.set_bad. GHAP uses <=0 or
    # NaN over sea; mirror that mask onto the prediction so both panels
    # show an identical white coastline.
    sea = ~np.isfinite(gt) | (gt <= 0.0)
    gt_m = np.ma.masked_where(sea, gt)
    pr_m = np.ma.masked_where(sea, np.clip(np.nan_to_num(pr), 0.0, None))

    land = ~sea
    if land.sum() > 0:
        rmse = float(np.sqrt(np.mean((pr[land] - gt[land]) ** 2)))
        mae = float(np.mean(np.abs(pr[land] - gt[land])))
        mean_gt = float(np.mean(gt[land]))
        vmax = max(40.0, float(np.percentile(gt[land], 98)))
    else:
        rmse = mae = mean_gt = float("nan")
        vmax = 40.0
    norm = Normalize(vmin=0, vmax=vmax)
    extent = win["extent"]

    n_panels = 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(11.0, 5.6),
        gridspec_kw={"wspace": 0.16},
    )
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")

    axes[0].imshow(
        gt_m, extent=extent, origin="upper", cmap=CMAP, norm=norm,
        aspect="equal", interpolation="nearest",
    )
    axes[0].set_title("GHAP truth", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Longitude (°E)", fontsize=9)
    axes[0].set_ylabel("Latitude (°N)", fontsize=9)

    im = axes[1].imshow(
        pr_m, extent=extent, origin="upper", cmap=CMAP, norm=norm,
        aspect="equal", interpolation="nearest",
    )
    axes[1].set_title(f"CRAN-PM T+1 — {date_str}", fontsize=11,
                      fontweight="bold")
    axes[1].set_xlabel("Longitude (°E)", fontsize=9)

    # ── EEA ground-truth station network overlay ──
    n_st = 0
    if show_stations:
        slon, slat = _stations_in_box(win)
        n_st = slon.size
        if n_st:
            for ax in axes:
                ax.scatter(
                    slon, slat, s=42, marker="o",
                    facecolors="none", edgecolors="#111111",
                    linewidths=1.1, zorder=5,
                )
            axes[0].scatter(
                [], [], s=42, marker="o", facecolors="none",
                edgecolors="#111111", linewidths=1.1,
                label=f"EEA PM$_{{2.5}}$ stations ({n_st})",
            )
            axes[0].legend(
                loc="upper left", fontsize=7.5, framealpha=0.9,
                handletextpad=0.4, borderpad=0.4,
            )
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.tick_params(labelsize=8)

    card = (
        f"RMSE = {rmse:.1f} µg m$^{{-3}}$\n"
        f"MAE  = {mae:.1f} µg m$^{{-3}}$\n"
        f"⟨GHAP⟩ = {mean_gt:.1f} µg m$^{{-3}}$"
    )
    if show_stations:
        card += f"\nEEA stations: {n_st}"
    axes[1].text(
        0.97, 0.04, card, transform=axes[1].transAxes, ha="right",
        va="bottom", fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                  boxstyle="round,pad=0.3"),
    )

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label(r"PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Pinpoint forecast — centre ({center_lat:.2f}°N, {center_lon:.2f}°E), "
        f"512×512 tile @ 0.01° native resolution",
        fontsize=10, y=0.99,
    )
    return fig


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2022-01-12")
    ap.add_argument("--lat", type=float, default=45.40)
    ap.add_argument("--lon", type=float, default=9.20)
    ap.add_argument("--out", default="/tmp/pinpoint_test.png")
    a = ap.parse_args()
    f = render_pinpoint(a.date, a.lat, a.lon)
    f.savefig(a.out, dpi=170, bbox_inches="tight")
    print(f"wrote {a.out}")
