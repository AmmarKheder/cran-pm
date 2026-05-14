#!/usr/bin/env python3
"""ECCV-style OOD visualisations for the 8 zero-shot regions.

For each region produces two figures, mirroring the ECCV `ood_teaser_*`
and `ood_scatter_*` pair:

  fig_ood_teaser_<region>.{pdf,png}
    Full-region map of the *hotspot day* (GHAP daily peak) with three
    zoomed-in 3x3 strips showing GT vs CRAN-PM at 1 km, the per-zoom
    RMSE in a small inset box, and a global RMSE annotation.

  fig_ood_scatter_<region>.{pdf,png}
    Hexbin scatter of annual-mean GHAP vs annual-mean CRAN-PM, with
    1:1 line, OLS fit, RMSE / Bias / R^2 / n annotation.

Data sources:
  predictions: evaluation_ood_v10f_otf/<region>/predictions_t1.zarr
  ground truth: ghap_global_daily/2022.zarr cropped on-the-fly to the
    region window declared in run_inference_ood_regions.py.

Sub-sampling: scatter plots use a factor-4 downsample (4192*6992 -> 1048*1748)
so the loop fits in a few minutes per region rather than ~20 minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import numpy as np
import zarr


# ── Region constants (identical to run_inference_ood_regions.py) ──────────────
GHAP_LAT_NORTH = {
    "india": 37.0, "usa_canada": 70.0, "egypt": 31.0, "chile": -17.0,
    "australia": -10.0, "china": 45.0, "southeast_asia": 22.0,
    "south_africa": -17.0,
}
GHAP_LON_WEST = {
    "india": 68.0, "usa_canada": -130.0, "egypt": 25.0, "chile": -76.0,
    "australia": 113.0, "china": 80.0, "southeast_asia": 98.0,
    "south_africa": 12.0,
}
GHAP_H, GHAP_W = 4192, 6992
GHAP_RES = 0.01
N_DAYS = 364   # pred[d] -> gt[d+1], d in 0..363

REGION_TITLE = {
    "india": "India", "usa_canada": "USA / Canada", "egypt": "Egypt",
    "chile": "Chile", "australia": "Australia", "china": "China",
    "southeast_asia": "Southeast Asia", "south_africa": "South Africa",
}
REGION_VMAX = {
    "india": 100, "usa_canada": 25, "egypt": 90, "chile": 50,
    "australia": 50, "china": 110, "southeast_asia": 70, "south_africa": 50,
}

# Three hotspot zooms per region — chosen for visual impact on annual
# composites. Each box is (lat_min, lat_max, lon_min, lon_max).
ZOOMS = {
    "india": [
        ("Delhi NCR",      28.0, 29.4,  76.5, 78.0),
        ("Indo-Gangetic",  25.0, 27.2,  80.8, 84.5),
        ("Mumbai",         18.6, 19.8,  72.4, 73.4),
    ],
    "usa_canada": [
        ("California Central Valley", 35.5, 38.0, -122.0, -120.0),
        ("Great Lakes",  41.5, 43.2, -88.0, -83.5),
        ("New York metro",40.4, 41.5, -74.6, -73.0),
    ],
    "china": [
        ("Beijing-Tianjin", 38.5, 41.2, 115.0, 118.0),
        ("Yangtze Delta",   30.0, 32.5, 119.5, 122.5),
        ("Sichuan Basin",   28.5, 31.5, 103.5, 106.5),
    ],
    "southeast_asia": [
        ("Bangkok",       13.2, 14.5, 100.0, 101.4),
        ("Jakarta",       -6.8, -5.6, 106.0, 107.4),
        ("Hanoi",         20.5, 21.7, 105.4, 106.4),
    ],
    "south_africa": [
        ("Johannesburg",  -27.0, -25.5,  27.5,  29.0),
        ("Cape Town",     -34.5, -33.5,  18.0,  19.2),
        ("Maputo",        -26.5, -25.5,  32.2,  33.2),
    ],
    "egypt": [
        ("Cairo",         29.5, 30.6, 30.8, 32.1),
        ("Nile Delta",    30.6, 31.6, 30.5, 31.6),
        ("Alexandria",    30.8, 31.4, 29.5, 30.4),
    ],
    "chile": [
        ("Santiago",      -33.9, -33.2, -71.0, -70.4),
        ("Concepcion",    -36.9, -36.4, -73.2, -72.7),
        ("Antofagasta",   -24.0, -23.4, -70.6, -70.1),
    ],
    "australia": [
        ("Sydney",        -34.4, -33.4, 150.5, 151.6),
        ("Melbourne",     -38.2, -37.4, 144.6, 145.7),
        ("Perth",         -32.4, -31.5, 115.7, 116.4),
    ],
}

DATA_GLOBAL_GHAP = Path("/scratch/project_462001140/ammar/eccv/data/zarr/"
                         "ghap_global_daily/2022.zarr")
EVAL_DIR = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/"
                "evaluation_ood_v10f_otf")

# GHAP-style colormap (used in the ECCV ood_teaser figures).
GHAP_COLORS = [
    "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
    "#9ecae1", "#c6dbef", "#deebf7",
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
]
CMAP = LinearSegmentedColormap.from_list("ghap", GHAP_COLORS, N=256)
CMAP.set_bad("white")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "axes.linewidth": 0.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _ghap_global_slice(day):
    """Open GHAP global zarr (365, 18000, 36000) lazily."""
    z = zarr.open(str(DATA_GLOBAL_GHAP), mode="r", zarr_format=2)
    return z[day]


def _crop_global_ghap(arr_global, region):
    lat0 = int(round((90.0 - GHAP_LAT_NORTH[region]) / GHAP_RES))
    lon0 = int(round((GHAP_LON_WEST[region] - (-180.0)) / GHAP_RES))
    return arr_global[lat0:lat0 + GHAP_H, lon0:lon0 + GHAP_W]


def region_extent(region):
    lat_n = GHAP_LAT_NORTH[region]
    lon_w = GHAP_LON_WEST[region]
    return (lon_w, lon_w + GHAP_W * GHAP_RES,
            lat_n - GHAP_H * GHAP_RES, lat_n)


# ── scatter (annual mean) ────────────────────────────────────────────────────
def make_scatter(region, out_dir):
    """Annual-mean GHAP vs CRAN-PM, hexbin density, down-sampled by 4 to fit memory/time."""
    pred_path = EVAL_DIR / region / "predictions_t1.zarr"
    if not pred_path.exists():
        print(f"[scatter {region}] predictions missing, skip")
        return

    pred_z = zarr.open(str(pred_path), mode="r", zarr_format=2)
    ghap_z = zarr.open(str(DATA_GLOBAL_GHAP), mode="r", zarr_format=2)
    lat0 = int(round((90.0 - GHAP_LAT_NORTH[region]) / GHAP_RES))
    lon0 = int(round((GHAP_LON_WEST[region] - (-180.0)) / GHAP_RES))
    lon_full_pix = 36000     # global GHAP lon dim
    wraps = (lon0 + GHAP_W) > lon_full_pix

    down = 4
    H, W = GHAP_H // down, GHAP_W // down
    g_sum = np.zeros((H, W), dtype=np.float64)
    p_sum = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.int32)

    def _crop_global(z, d):
        """Crop a day from the global GHAP zarr, handling antimeridian wrap."""
        if not wraps:
            return np.asarray(z[d, lat0:lat0 + GHAP_H,
                                lon0:lon0 + GHAP_W], dtype=np.float32)
        right = np.asarray(z[d, lat0:lat0 + GHAP_H,
                              lon0:lon_full_pix], dtype=np.float32)
        left = np.asarray(z[d, lat0:lat0 + GHAP_H,
                             0:lon0 + GHAP_W - lon_full_pix], dtype=np.float32)
        return np.concatenate([right, left], axis=1)

    print(f"[scatter {region}] accumulating {N_DAYS} days "
          f"({H}x{W} downsampled by {down}, wraps={wraps}) ...",
          flush=True)
    for d in range(N_DAYS):
        gt_full = _crop_global(ghap_z, d + 1)
        pr_full = np.asarray(pred_z[d], dtype=np.float32)
        # Box-down-sample (truncate to multiple of `down`).
        gt = gt_full[:H * down, :W * down].reshape(
            H, down, W, down).mean(axis=(1, 3))
        pr = pr_full[:H * down, :W * down].reshape(
            H, down, W, down).mean(axis=(1, 3))
        pr = np.nan_to_num(np.clip(pr, 0.0, None), nan=0.0)
        valid = np.isfinite(gt) & (gt > 0.5) & np.isfinite(pr)
        g_sum[valid] += gt[valid]
        p_sum[valid] += pr[valid]
        cnt[valid] += 1
        if d % 90 == 0:
            print(f"  day {d}/{N_DAYS}", flush=True)

    land = cnt > (N_DAYS // 2)
    gm = (g_sum[land] / cnt[land]).astype(np.float32)
    pm = (p_sum[land] / cnt[land]).astype(np.float32)
    # Drop residual NaNs (predictions outside tile coverage at region edges).
    finite = np.isfinite(gm) & np.isfinite(pm)
    gm = gm[finite]; pm = pm[finite]
    rmse = float(np.sqrt(np.mean((pm - gm) ** 2)))
    bias = float(np.mean(pm - gm))
    r = float(np.corrcoef(gm, pm)[0, 1]) if len(gm) > 10 else float("nan")
    r2 = r * r if np.isfinite(r) else float("nan")
    n = int(finite.sum())
    vmax = REGION_VMAX[region] * 1.05
    print(f"  RMSE={rmse:.2f} bias={bias:+.2f} R^2={r2:.3f} n={n:,}",
          flush=True)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    hb = ax.hexbin(gm, pm, gridsize=80, cmap="RdYlBu_r",
                   mincnt=1, bins="log",
                   extent=[0, vmax, 0, vmax], linewidths=0.05)
    cb = fig.colorbar(hb, ax=ax, label=r"$\log_{10}$(pixel count)")
    cb.ax.tick_params(labelsize=8)
    ax.plot([0, vmax], [0, vmax], "k--", lw=1.2, label="1:1")
    m, b = np.polyfit(gm, pm, 1)
    xs = np.array([0, vmax])
    ax.plot(xs, m * xs + b, color="steelblue", lw=1.2,
            label=f"fit: y = {m:.2f}x{b:+.1f}")
    ax.set_xlabel(r"Annual-mean GHAP PM$_{2.5}$ ($\mu$g m$^{-3}$)",
                  fontsize=10)
    ax.set_ylabel(r"Annual-mean CRAN-PM PM$_{2.5}$ ($\mu$g m$^{-3}$)",
                  fontsize=10)
    ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
    ax.set_aspect("equal")
    ax.text(0.97, 0.04,
            f"RMSE = {rmse:.2f} µg/m³\n"
            f"Bias = {bias:+.2f} µg/m³\n"
            f"$R^2$  = {r2:.3f}\n"
            f"$n$    = {n:,} pixels",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                      boxstyle="round,pad=0.4"))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.set_title(f"CRAN-PM zero-shot — {REGION_TITLE[region]} | 2022 annual mean",
                  fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)

    out_pdf = out_dir / f"fig_ood_scatter_{region}.pdf"
    out_png = out_dir / f"fig_ood_scatter_{region}.png"
    fig.savefig(str(out_pdf), bbox_inches="tight")
    fig.savefig(str(out_png), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {out_pdf} and .png", flush=True)
    return {"rmse": rmse, "bias": bias, "r2": r2, "n": n}


# ── teaser (hotspot day) ──────────────────────────────────────────────────────
def _crop_ghap_day(ghap_z, day, region):
    """Crop a single day of GHAP for a region, handling antimeridian wrap."""
    lat0 = int(round((90.0 - GHAP_LAT_NORTH[region]) / GHAP_RES))
    lon0 = int(round((GHAP_LON_WEST[region] - (-180.0)) / GHAP_RES))
    lon_full = ghap_z.shape[2]
    if lon0 + GHAP_W <= lon_full:
        return np.asarray(ghap_z[day, lat0:lat0 + GHAP_H,
                                  lon0:lon0 + GHAP_W], dtype=np.float32)
    right = np.asarray(ghap_z[day, lat0:lat0 + GHAP_H,
                               lon0:lon_full], dtype=np.float32)
    left = np.asarray(ghap_z[day, lat0:lat0 + GHAP_H,
                              0:lon0 + GHAP_W - lon_full], dtype=np.float32)
    return np.concatenate([right, left], axis=1)


def find_hotspot_day(region):
    """Day with highest 99th-percentile GT PM2.5 over the region (downsampled scan)."""
    ghap_z = zarr.open(str(DATA_GLOBAL_GHAP), mode="r", zarr_format=2)
    best_d, best_v = 0, -1.0
    for d in range(N_DAYS):
        s = _crop_ghap_day(ghap_z, d, region)[::50, ::50]
        v = float(np.percentile(s, 99))
        if v > best_v:
            best_v = v; best_d = d
    return best_d, best_v


def make_teaser(region, out_dir):
    pred_path = EVAL_DIR / region / "predictions_t1.zarr"
    if not pred_path.exists():
        print(f"[teaser {region}] predictions missing, skip")
        return
    print(f"[teaser {region}] finding hotspot day ...", flush=True)
    d_hot, v_hot = find_hotspot_day(region)
    print(f"  hotspot day-of-year = {d_hot}, p99 = {v_hot:.1f}", flush=True)

    pred_z = zarr.open(str(pred_path), mode="r", zarr_format=2)
    ghap_z = zarr.open(str(DATA_GLOBAL_GHAP), mode="r", zarr_format=2)

    gt = _crop_ghap_day(ghap_z, d_hot + 1, region)
    pr = np.nan_to_num(
        np.asarray(pred_z[d_hot], dtype=np.float32), nan=0.0)
    pr = np.clip(pr, 0.0, None)
    finite = np.isfinite(gt) & np.isfinite(pr)
    rmse = float(np.sqrt(np.mean((pr[finite] - gt[finite]) ** 2)))

    extent = region_extent(region)
    vmax = REGION_VMAX[region]
    norm = Normalize(vmin=0, vmax=vmax)
    zooms = ZOOMS.get(region, [])[:3]

    fig = plt.figure(figsize=(14.5, 6.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[2.8, 1.0, 1.0, 0.04],
                           height_ratios=[1, 1, 1],
                           wspace=0.05, hspace=0.05)
    # Main map.
    ax_main = fig.add_subplot(gs[:, 0])
    # Downsample for plotting speed.
    sub = 8
    im = ax_main.imshow(gt[::sub, ::sub], extent=extent, origin="upper",
                        cmap=CMAP, norm=norm, aspect="auto",
                        interpolation="bilinear")
    ax_main.set_xlabel("Longitude", fontsize=10)
    ax_main.set_ylabel("Latitude", fontsize=10)
    ax_main.set_title(f"(a) Ground truth (GHAP)",
                      fontsize=11, fontweight="bold", loc="left")
    ax_main.text(0.97, 0.04,
                  f"{2022}-{d_hot+1:03d} (DOY)\nRMSE = {rmse:.2f} µg/m³",
                  transform=ax_main.transAxes, ha="right", va="bottom",
                  fontsize=9,
                  bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                            boxstyle="round,pad=0.3"))
    ax_main.tick_params(labelsize=8.5)

    # Zoom rectangles on main map + companion zoom panels.
    for k, (zname, lat_min, lat_max, lon_min, lon_max) in enumerate(zooms):
        rect = mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            edgecolor="red", facecolor="none", lw=1.4)
        ax_main.add_patch(rect)
        ax_main.text((lon_min + lon_max) / 2, lat_max + 0.2,
                      f"({chr(98 + k)})", color="red", fontsize=10,
                      fontweight="bold", ha="center")

        # row k → 2 zoom panels (GT, Pred).
        r0 = int(round((GHAP_LAT_NORTH[region] - lat_max) / GHAP_RES))
        r1 = int(round((GHAP_LAT_NORTH[region] - lat_min) / GHAP_RES))
        c0 = int(round((lon_min - GHAP_LON_WEST[region]) / GHAP_RES))
        c1 = int(round((lon_max - GHAP_LON_WEST[region]) / GHAP_RES))
        r0, r1 = max(0, r0), min(GHAP_H, r1)
        c0, c1 = max(0, c0), min(GHAP_W, c1)
        gt_z = gt[r0:r1, c0:c1]
        pr_z = pr[r0:r1, c0:c1]
        rmse_z = float(np.sqrt(np.mean((pr_z - gt_z) ** 2)))

        ax_gt = fig.add_subplot(gs[k, 1])
        ax_gt.imshow(gt_z, cmap=CMAP, norm=norm,
                     extent=(lon_min, lon_max, lat_min, lat_max),
                     origin="upper", interpolation="bilinear")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        ax_gt.text(0.04, 0.96, f"({chr(98 + k)}) {zname}",
                   transform=ax_gt.transAxes, ha="left", va="top",
                   fontsize=8.5, color="white",
                   bbox=dict(facecolor="black", alpha=0.55,
                             edgecolor="none", pad=1.5))
        if k == 0:
            ax_gt.set_title("GT (zoom)", fontsize=10, fontweight="bold")

        ax_pr = fig.add_subplot(gs[k, 2])
        ax_pr.imshow(pr_z, cmap=CMAP, norm=norm,
                     extent=(lon_min, lon_max, lat_min, lat_max),
                     origin="upper", interpolation="bilinear")
        ax_pr.set_xticks([]); ax_pr.set_yticks([])
        ax_pr.text(0.96, 0.04, f"RMSE = {rmse_z:.2f}",
                    transform=ax_pr.transAxes, ha="right", va="bottom",
                    fontsize=8, color="white",
                    bbox=dict(facecolor="black", alpha=0.55,
                              edgecolor="none", pad=1.5))
        if k == 0:
            ax_pr.set_title("CRAN-PM T+1", fontsize=10, fontweight="bold")

    # Shared colorbar on the right.
    cax = fig.add_subplot(gs[:, 3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP),
                       cax=cax)
    cb.set_label(r"PM$_{2.5}$ ($\mu$g m$^{-3}$)", fontsize=10)
    cb.ax.tick_params(labelsize=8.5)

    fig.suptitle(
        f"CRAN-PM zero-shot — {REGION_TITLE[region]}  |  "
        f"2022-DOY {d_hot+1} (hotspot)",
        fontsize=13, fontweight="bold", y=0.97,
    )
    out_pdf = out_dir / f"fig_ood_teaser_{region}.pdf"
    out_png = out_dir / f"fig_ood_teaser_{region}.png"
    fig.savefig(str(out_pdf), bbox_inches="tight")
    fig.savefig(str(out_png), bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(f"  wrote {out_pdf} and .png", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+",
                        default=list(REGION_TITLE.keys()))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/scratch/project_462001140/ammar/"
                                     "eccv/cran-pm/paper/figures"))
    parser.add_argument("--no-scatter", action="store_true")
    parser.add_argument("--no-teaser", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for r in args.regions:
        print(f"\n========== {r.upper()} ==========")
        if not args.no_scatter:
            stats = make_scatter(r, args.out_dir)
            if stats is not None:
                summary[r] = stats
        if not args.no_teaser:
            make_teaser(r, args.out_dir)

    if summary:
        sp = args.out_dir / "ood_viz_stats.json"
        with open(sp, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote summary to {sp}")


if __name__ == "__main__":
    main()
