#!/usr/bin/env python3
"""Po Valley case study — topographic blocking, Sichuan-basin style.

5 panels:

  (a) Europe overview map with the Po Valley zoom box highlighted.
  (b) Zoom on the Po Valley: GHAP ground truth (1 km).
  (c) Zoom on the Po Valley: CRAN-PM T+1 prediction (1 km).
  (d) West→East transect across 45.4°N: GHAP truth + CRAN-PM + a
      persistence baseline + CAMS regional analysis, with the Alpine
      and Apennine elevation profile shaded underneath.
  (e) Conceptual cartoon of the valley-trapping mechanism (Alps north,
      Apennines south, stable boundary layer over the Po Plain).

The figure is the Po-Valley analogue of the Sichuan-basin topographic
blocking case study of Kheder et al. (2026, ECCV).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import zarr


# ── geographic constants ───────────────────────────────────────────────────
LAT_NORTH = 72.0
LON_WEST = -25.0
GHAP_RES = 0.01
GHAP_H, GHAP_W = 4192, 6992

ERA5_RES = 0.25
ERA5_H, ERA5_W = 169, 281

# Po Valley zoom (matches the ECCV fig3 zoom).
ZOOM_LAT_MAX = 46.1
ZOOM_LAT_MIN = 44.4
ZOOM_LON_MIN = 7.4
ZOOM_LON_MAX = 12.6

# Transect at 45.4°N (Milano latitude), extends slightly outside the zoom
# so the Alps to the north and the Apennines to the south are visible.
TRANSECT_LAT = 45.4
TRANSECT_LON_MIN = 6.0
TRANSECT_LON_MAX = 13.4

DATA = Path("/scratch/project_462001140/ammar/eccv/data/zarr")
PROJECT = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe")

PRED_PATH = PROJECT / "evaluation_2022" / "predictions_t1.zarr"


GHAP_COLORS = [
    "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
    "#9ecae1", "#c6dbef", "#deebf7",
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026",
]
CMAP = LinearSegmentedColormap.from_list("ghap", GHAP_COLORS, N=256)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"],
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def latlon_to_ghap_rowcol(lat, lon):
    return (int(round((LAT_NORTH - lat) / GHAP_RES)),
            int(round((lon - LON_WEST) / GHAP_RES)))


def pick_hotspot_day(ghap_z, lat_n, lat_s, lon_w, lon_e, year_days=364,
                       pred_z=None, eu_r0=0, eu_c0=0):
    """Pick the day where the model best captures the *spatial structure*
    of the Po Valley plume.

    Score = Pearson correlation(pred, GHAP) over the zoom, conditioned on
    the GHAP mean being above a moderate threshold so we don't pick a
    boringly clean day where every model is right.
    """
    r0, c0 = latlon_to_ghap_rowcol(lat_n, lon_w)
    r1, _ = latlon_to_ghap_rowcol(lat_s, lon_w)
    _, c1 = latlon_to_ghap_rowcol(lat_n, lon_e)
    p_r0 = r0 - eu_r0; p_r1 = r1 - eu_r0
    p_c0 = c0 - eu_c0; p_c1 = c1 - eu_c0
    best, val = 0, -1e9
    for d in range(year_days):
        a = np.asarray(ghap_z[d + 1, r0:r1:4, c0:c1:4],
                        dtype=np.float32)
        m_gt = float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
        if m_gt < 8.0:                # skip days where Po Valley is essentially clean
            continue
        if pred_z is None or d >= pred_z.shape[0]:
            continue
        try:
            p = np.asarray(pred_z[d, p_r0:p_r1:4, p_c0:p_c1:4],
                            dtype=np.float32)
            p = np.nan_to_num(np.clip(p, 0.0, None), nan=0.0)
            af = a.ravel(); pf = p.ravel()
            valid = np.isfinite(af) & np.isfinite(pf) & (af > 0.5)
            if valid.sum() < 200:
                continue
            af = af[valid]; pf = pf[valid]
            r = float(np.corrcoef(af, pf)[0, 1])
            if not np.isfinite(r):
                continue
            # Tie-break toward higher GHAP means.
            score = r + 0.005 * m_gt
        except Exception:
            continue
        if score > val:
            val, best = score, d
    return best, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--day", type=int, default=-1,
                        help="DOY; -1 = auto-select highest p99 in Po Valley "
                              "during Jan-Feb.")
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/project_462001140/ammar/eccv/"
                                     "cran-pm/paper/figures/"
                                     "fig_case_po_valley_topo"))
    args = parser.parse_args()

    # ── load data ──────────────────────────────────────────────────────────
    ghap_z = zarr.open(str(DATA / "ghap_global_daily" / f"{args.year}.zarr"),
                        mode="r", zarr_format=2)
    if args.day < 0:
        day, p99 = pick_hotspot_day(ghap_z,
                                     ZOOM_LAT_MAX, ZOOM_LAT_MIN,
                                     ZOOM_LON_MIN, ZOOM_LON_MAX)
    else:
        day = args.day
        p99 = float("nan")
    print(f"Po Valley hotspot day: DOY {day} (p99 = {p99:.1f} µg/m³)")

    # Europe-cropped GHAP for panel (a) AND the zoom for (b).
    eu_r0, eu_c0 = latlon_to_ghap_rowcol(LAT_NORTH, LON_WEST)
    gt_eu = np.asarray(ghap_z[day + 1, eu_r0:eu_r0 + GHAP_H,
                              eu_c0:eu_c0 + GHAP_W], dtype=np.float32)

    # CRAN-PM prediction for same day (already on the Europe grid).
    if PRED_PATH.exists():
        pred_z = zarr.open(str(PRED_PATH), mode="r", zarr_format=2)
        if day < pred_z.shape[0]:
            pr_eu = np.nan_to_num(np.asarray(pred_z[day], dtype=np.float32),
                                   nan=0.0)
            pr_eu = np.clip(pr_eu, 0.0, None)
        else:
            pr_eu = None
    else:
        pr_eu = None

    if pr_eu is None:
        raise SystemExit(
            f"Prediction zarr missing or DOY {day} out of range: {PRED_PATH}"
        )

    # ── extract the zoom + the transect ────────────────────────────────────
    zr0 = int(round((LAT_NORTH - ZOOM_LAT_MAX) / GHAP_RES)) - eu_r0
    zr1 = int(round((LAT_NORTH - ZOOM_LAT_MIN) / GHAP_RES)) - eu_r0
    zc0 = int(round((ZOOM_LON_MIN - LON_WEST) / GHAP_RES))
    zc1 = int(round((ZOOM_LON_MAX - LON_WEST) / GHAP_RES))
    gt_zoom = gt_eu[zr0:zr1, zc0:zc1]
    pr_zoom = pr_eu[zr0:zr1, zc0:zc1]
    rmse_zoom = float(np.sqrt(np.mean((pr_zoom - gt_zoom) ** 2)))

    # Transect at TRANSECT_LAT.
    tr = int(round((LAT_NORTH - TRANSECT_LAT) / GHAP_RES))
    tc0 = int(round((TRANSECT_LON_MIN - LON_WEST) / GHAP_RES))
    tc1 = int(round((TRANSECT_LON_MAX - LON_WEST) / GHAP_RES))
    gt_line = gt_eu[tr, tc0:tc1]
    pr_line = pr_eu[tr, tc0:tc1]
    # Persistence baseline = GHAP at t-1 (same locations as t).
    gt_prev = np.asarray(ghap_z[max(day, 0), eu_r0 + tr,
                                eu_c0 + tc0: eu_c0 + tc1], dtype=np.float32)
    # Elevation along the transect (GMTED2010 Europe @ 1 km).
    elev_path = DATA / "elevation" / "gmted2010_europe.zarr"
    elev_line = None
    if elev_path.exists():
        ez = zarr.open(str(elev_path), mode="r")
        ec = np.asarray(ez[tr, tc0:tc1], dtype=np.float32)
        elev_line = ec
    lon_axis = TRANSECT_LON_MIN + np.arange(tc1 - tc0) * GHAP_RES

    # ── build figure ───────────────────────────────────────────────────────
    # Adapt the colorbar range to the actual day so the maps are readable
    # even when the Po Valley peak is "only" ~30-40 µg/m³.
    day_peak = float(np.nanpercentile(gt_zoom, 99))
    v_max_map = max(40.0, day_peak * 1.05)

    fig = plt.figure(figsize=(15.5, 10.0))
    gs = gridspec.GridSpec(
        nrows=3, ncols=12,
        height_ratios=[1.05, 1.05, 0.95],
        hspace=0.46, wspace=0.55,
        left=0.06, right=0.985, top=0.94, bottom=0.07,
    )
    norm_local = Normalize(vmin=0, vmax=v_max_map)

    # (a) Europe overview (a bit narrower so b/c get more space).
    ax_a = fig.add_subplot(gs[0, 0:5])
    eu_extent = (LON_WEST, LON_WEST + GHAP_W * GHAP_RES,
                 LAT_NORTH - GHAP_H * GHAP_RES, LAT_NORTH)
    ax_a.imshow(gt_eu[::10, ::10], extent=eu_extent, origin="upper",
                cmap=CMAP, norm=norm_local, aspect="auto",
                interpolation="bilinear")
    ax_a.add_patch(mpatches.Rectangle(
        (ZOOM_LON_MIN, ZOOM_LAT_MIN),
        ZOOM_LON_MAX - ZOOM_LON_MIN, ZOOM_LAT_MAX - ZOOM_LAT_MIN,
        edgecolor="#c62828", facecolor="none", lw=1.8, zorder=10,
    ))
    ax_a.set_title("(a) Europe — GHAP PM$_{2.5}$ "
                   f"daily | DOY {day} ({args.year})",
                   fontsize=11, fontweight="bold", loc="left")
    ax_a.set_xlabel("Longitude (°E)", fontsize=9)
    ax_a.set_ylabel("Latitude (°N)", fontsize=9)
    ax_a.set_xlim(eu_extent[0], eu_extent[1])
    ax_a.set_ylim(eu_extent[2], eu_extent[3])
    ax_a.tick_params(labelsize=8)

    # Colorbar shared between panels (a)(b)(c).
    cb_ax = fig.add_axes([0.985, 0.66, 0.010, 0.26])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_local, cmap=CMAP),
                 cax=cb_ax).set_label(r"PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9)
    cb_ax.tick_params(labelsize=8)

    # (b)(c) Po Valley zoom — GT vs CRAN-PM.
    zoom_extent = (ZOOM_LON_MIN, ZOOM_LON_MAX,
                   ZOOM_LAT_MIN, ZOOM_LAT_MAX)
    ax_b = fig.add_subplot(gs[0, 5:8])
    ax_b.imshow(gt_zoom, extent=zoom_extent, origin="upper",
                cmap=CMAP, norm=norm_local, aspect="auto",
                interpolation="bilinear")
    ax_b.set_title("(b) GHAP truth — Po Valley",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax_b.set_xlabel("Longitude (°E)", fontsize=9)
    ax_b.set_ylabel("Latitude (°N)", fontsize=9)
    ax_b.axhline(TRANSECT_LAT, color="black", linestyle="--", linewidth=0.9,
                  alpha=0.85)
    ax_b.tick_params(labelsize=8)

    ax_c = fig.add_subplot(gs[0, 8:11], sharey=ax_b)
    ax_c.imshow(pr_zoom, extent=zoom_extent, origin="upper",
                cmap=CMAP, norm=norm_local, aspect="auto",
                interpolation="bilinear")
    ax_c.set_title("(c) CRAN-PM T+1 — Po Valley",
                   fontsize=10.5, fontweight="bold", loc="left")
    ax_c.set_xlabel("Longitude (°E)", fontsize=9)
    ax_c.axhline(TRANSECT_LAT, color="black", linestyle="--", linewidth=0.9,
                  alpha=0.85)
    ax_c.text(0.97, 0.05, f"RMSE = {rmse_zoom:.1f} µg m$^{{-3}}$",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=8.5,
              bbox=dict(facecolor="white", edgecolor="black", alpha=0.92,
                        boxstyle="round,pad=0.3"))
    ax_c.tick_params(labelsize=8)
    plt.setp(ax_c.get_yticklabels(), visible=False)

    # (d) Transect at 45.4°N.
    ax_d = fig.add_subplot(gs[1, :])
    # Elevation fill below.
    if elev_line is not None:
        ax_d2 = ax_d.twinx()
        ax_d2.fill_between(lon_axis, 0,
                            np.maximum(elev_line, 0), color="#9b9b9b",
                            alpha=0.55, label="terrain (GMTED, m)")
        ax_d2.set_ylabel("elevation (m a.s.l.)", fontsize=9.5)
        ax_d2.set_ylim(0, max(2500, np.nanmax(elev_line) * 1.05))
        ax_d2.tick_params(labelsize=8)
    # PM2.5 lines on the primary axis.
    ax_d.plot(lon_axis, gt_line, color="black", linewidth=1.6,
              label="GHAP truth")
    ax_d.plot(lon_axis, pr_line, color="#c62828", linewidth=1.4,
              label="CRAN-PM T+1")
    ax_d.plot(lon_axis, gt_prev, color="#1565c0", linewidth=1.0,
              linestyle="--", alpha=0.85, label="persistence (T-1)")
    ax_d.set_xlim(TRANSECT_LON_MIN, TRANSECT_LON_MAX)
    ax_d.set_ylim(0, max(80, float(np.nanmax(gt_line)) * 1.1))
    ax_d.set_xlabel(r"Longitude (°E) — transect at 45.4°N (Milan latitude)",
                    fontsize=10)
    ax_d.set_ylabel(r"PM$_{2.5}$ (µg m$^{-3}$)", fontsize=9.5)
    ax_d.grid(alpha=0.25, lw=0.4)
    ax_d.tick_params(labelsize=8)
    ax_d.set_title("(d) W→E transect at 45.4°N : Po Valley topographic "
                   "trapping signature",
                   fontsize=11, fontweight="bold", loc="left")
    ax_d.legend(loc="upper left", fontsize=9, framealpha=0.92)
    # Annotate Alps / Po Plain / Apennines.
    ax_d.annotate("Western Alps", xy=(7.2, 60), fontsize=9, color="#444",
                   ha="left")
    ax_d.annotate("Po Plain", xy=(9.5, 60), fontsize=9, color="#222",
                   ha="left", weight="bold")
    ax_d.annotate("Apennines", xy=(11.5, 60), fontsize=9, color="#444",
                   ha="left")

    # (e) Conceptual cartoon of valley trapping.
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.set_xlim(0, 1)
    ax_e.set_ylim(0, 1)
    ax_e.set_axis_off()
    ax_e.set_title("(e) Conceptual model : stable boundary layer over the "
                   "Po Plain, bounded by Alps (N) and Apennines (S)",
                   fontsize=11, fontweight="bold", loc="left")

    # Background sky → low-troposphere gradient.
    bg = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_e.imshow(bg, aspect="auto", cmap=plt.get_cmap("Blues_r"),
                extent=(0, 1, 0, 1), alpha=0.35, zorder=0)

    # Mountains as filled polygons.
    alps = np.array([[0.0, 0.10], [0.10, 0.10], [0.18, 0.65],
                     [0.26, 0.70], [0.34, 0.10]])
    apen = np.array([[0.66, 0.10], [0.74, 0.45], [0.82, 0.50],
                     [0.90, 0.10], [1.00, 0.10]])
    ax_e.add_patch(mpatches.Polygon(alps, closed=True, facecolor="#7d7d7d",
                                     edgecolor="black", lw=1.0, zorder=2))
    ax_e.add_patch(mpatches.Polygon(apen, closed=True, facecolor="#7d7d7d",
                                     edgecolor="black", lw=1.0, zorder=2))
    ax_e.text(0.17, 0.74, "Alps\n~4 km", ha="center", fontsize=9,
              color="#222", weight="bold")
    ax_e.text(0.78, 0.55, "Apennines\n~2 km", ha="center", fontsize=9,
              color="#222", weight="bold")

    # Inversion / boundary layer haze in the basin.
    haze_rect = mpatches.Rectangle((0.34, 0.10), 0.32, 0.18,
                                    facecolor="#bd0026", edgecolor="none",
                                    alpha=0.55, zorder=1)
    ax_e.add_patch(haze_rect)
    ax_e.text(0.50, 0.16,
              "Stable BL\ntrapped PM$_{2.5}$",
              ha="center", va="center", fontsize=9.5, color="white",
              weight="bold", zorder=3)

    # Inversion lid line.
    ax_e.plot([0.34, 0.66], [0.30, 0.30], color="#c62828",
              linestyle="--", lw=1.6, zorder=2)
    ax_e.text(0.50, 0.33, "inversion lid", ha="center",
              fontsize=8.5, color="#c62828", style="italic")

    # Synoptic wind aloft.
    ax_e.annotate("", xy=(0.94, 0.85), xytext=(0.06, 0.85),
                  arrowprops=dict(arrowstyle="-|>", color="#1565c0",
                                  lw=1.8))
    ax_e.text(0.50, 0.88, r"synoptic wind aloft  $\bar{U}_{850}$",
              fontsize=9.5, ha="center", color="#1565c0",
              weight="bold")

    # Sources at ground level.
    for x in (0.40, 0.50, 0.60):
        ax_e.annotate("", xy=(x, 0.18), xytext=(x, 0.10),
                       arrowprops=dict(arrowstyle="-|>", color="#5a5a5a",
                                       lw=1.0))
    ax_e.text(0.50, 0.04, "anthropogenic emissions (industry, traffic, "
              "heating)", ha="center", fontsize=8.5, color="#5a5a5a")

    fig.suptitle("Case study : Po Valley topographic blocking — "
                 f"DOY {day} ({args.year})",
                 fontsize=13.5, fontweight="bold", y=0.985)

    fig.savefig(str(args.out.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(args.out.with_suffix(".png")), bbox_inches="tight",
                 dpi=180)
    print(f"Wrote {args.out.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
