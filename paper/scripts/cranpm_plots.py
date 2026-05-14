"""Reusable plotting helpers for the CRAN-PM GMD paper.

All figures produced for the paper go through this module. Centralising
the styling (fonts, colors, projections, colorbar conventions) makes the
figure set look like one paper rather than a patchwork.

Projections used:
    Robinson  -> global context (study-domain figure, OOD overview).
    LAEA      -> Europe regional maps (EPSG:3035, official EU projection).
    PlateCarree -> raw data display when grid alignment matters.
"""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PAPER_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.7,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Method palette — matches the ECCV CRAN-PM paper (Kheder et al. 2026).
METHOD_COLORS = {
    "CRAN-PM": "#D62728",   # signature red, thick lines + filled markers
    "TopoFlow": "#A0522D",  # brown
    "ClimaX":   "#9467BD",  # purple
    "ConvLSTM": "#FF9933",  # orange
    "SimVP":    "#56B4E9",  # light blue
    "Earthformer": "#009E73",  # emerald green
    "Persistence": "#7F7F7F",
    "CAMS":     "#BCBD22",
    "Aurora":   "#17BECF",
}

METHOD_MARKERS = {
    "CRAN-PM": "*",      # star, large
    "TopoFlow": "s",     # square
    "ClimaX": "o",
    "ConvLSTM": "D",
    "SimVP": "^",
    "Earthformer": "v",
    "Persistence": ".",
    "CAMS": "P",
    "Aurora": "X",
}

# ECCV charter: RdYlBu_r for PM2.5 (intuitive blue=clean, red=polluted).
CMAP_PM = "RdYlBu_r"
CMAP_BIAS = "RdBu_r"
CMAP_RMSE = "viridis"


def apply_paper_style() -> None:
    mpl.rcParams.update(PAPER_STYLE)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# GHAP Europe grid (4192, 6992) at 0.01 deg.
GHAP_LAT_NORTH = 72.0
GHAP_LON_WEST = -25.0
GHAP_RES_DEG = 0.01


def ghap_lat_lon():
    nlat, nlon = 4192, 6992
    lats = GHAP_LAT_NORTH - GHAP_RES_DEG * (np.arange(nlat) + 0.5)
    lons = GHAP_LON_WEST + GHAP_RES_DEG * (np.arange(nlon) + 0.5)
    return lats, lons


def europe_extent_pc():
    # Plate Carree extent (W, E, S, N). Used for all Europe maps regardless of
    # display projection — cartopy converts from PC to the target CRS.
    return [-25.0, 45.0, 30.0, 72.0]


# ---------------------------------------------------------------------------
# Map factories
# ---------------------------------------------------------------------------

def make_robinson_axis(ax=None, central_longitude: float = 10.0):
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(10, 5),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=central_longitude)},
        )
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f1f1f1", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f0fa", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#666666", linewidth=0.4, zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor="#888888", linewidth=0.25, zorder=2)
    return ax


def make_europe_axis(ax=None, projection: str = "pc"):
    """Make a Europe-extent axis. Default `pc` (PlateCarree) matches the
    ECCV CRAN-PM paper's figure style; `laea` is also available."""
    proj_map = {
        "pc": ccrs.PlateCarree(),
        "laea": ccrs.LambertAzimuthalEqualArea(central_longitude=10.0, central_latitude=52.0),
        "lcc": ccrs.LambertConformal(central_longitude=10.0, central_latitude=52.0),
    }
    proj = proj_map[projection]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.6), subplot_kw={"projection": proj})
    ax.set_extent(europe_extent_pc(), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#dde8f3", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f7f3ee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#2b2b2b", linewidth=0.45, zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor="#555555", linewidth=0.3,
                   linestyle="-", zorder=2)
    return ax


def add_colorbar(ax, mappable, label: str, orientation: str = "horizontal"):
    cbar = plt.colorbar(
        mappable,
        ax=ax,
        orientation=orientation,
        shrink=0.78,
        pad=0.05,
        aspect=30,
        extend="both",
    )
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.4)
    return cbar


def add_stat_box(ax, lines: list[str], loc: str = "lower left",
                 facecolor: str = "white", alpha: float = 0.78) -> None:
    """ECCV-style stat overlay box (RMSE / MAE / R) in a corner of a map."""
    from matplotlib.offsetbox import AnchoredText
    text = "\n".join(lines)
    at = AnchoredText(
        text, loc=loc, frameon=True, prop=dict(size=8, family="DejaVu Sans"),
        pad=0.35, borderpad=0.4,
    )
    at.patch.set_facecolor(facecolor)
    at.patch.set_alpha(alpha)
    at.patch.set_edgecolor("#555555")
    at.patch.set_linewidth(0.4)
    ax.add_artist(at)


def panel_label(ax, label: str, loc=(0.02, 0.97)) -> None:
    """ECCV-style bold panel label like '(a) Ground Truth (GHAP)'."""
    ax.text(loc[0], loc[1], label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, pad=2))


# ---------------------------------------------------------------------------
# Data IO helpers
# ---------------------------------------------------------------------------

EVAL_BASE = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/evaluation_2022")


def load_annual_means():
    d = np.load(EVAL_BASE / "annual_mean_t1.npz")
    return d["gt_mean"], d["pred_mean"], d["count"]


def load_best_day():
    d = np.load(EVAL_BASE / "best_day_maps.npz", allow_pickle=True)
    return {
        "date": str(d["date"]),
        "gt_t1": d["gt_t1"],
        "pred_t1": d["pred_t1"],
        "gt_t2": d["gt_t2"],
        "pred_t2": d["pred_t2"],
    }


def load_method_metrics():
    """Return dict[method_label] -> {T+lt: {1km: {...}, 25km: {...}}}."""
    import json

    methods = {
        "CRAN-PM": "cranpm",
        "TopoFlow": "topoflow",
        "ClimaX": "climax",
        "ConvLSTM": "convlstm",
        "Earthformer": "earthformer",
        "SimVP": "simvp",
        "CAMS": "cams",
        "Persistence": "persistence",
    }
    out = {}
    for label, slug in methods.items():
        path = EVAL_BASE / f"main_table_{slug}.json"
        if path.exists():
            with open(path) as fh:
                out[label] = json.load(fh)
    return out


def load_country_mae():
    d = np.load(EVAL_BASE / "country_mae_cache.npz", allow_pickle=True)
    return d["mae_results"].item()


def load_global_validation_2025():
    """2.8M rows of EEA + global station observations (2025)."""
    return np.load("/pfs/lustrep1/scratch/project_462000640/ammar/Master_Validation_2025.npz")


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample for fast plotting at full Europe extent."""
    h, w = arr.shape
    h2, w2 = h // factor, w // factor
    cropped = arr[: h2 * factor, : w2 * factor]
    return cropped.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Output management
# ---------------------------------------------------------------------------

PAPER_FIG_DIR = Path("/scratch/project_462001140/ammar/eccv/cran-pm/paper/figures")
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, slug: str, formats=("pdf", "png")) -> list[Path]:
    """Save a figure as both PDF (paper) and PNG (preview)."""
    paths = []
    for ext in formats:
        out = PAPER_FIG_DIR / f"{slug}.{ext}"
        fig.savefig(out)
        paths.append(out)
    return paths
