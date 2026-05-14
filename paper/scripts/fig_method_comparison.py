"""Figure: method comparison — RMSE / R² per lead time, ECCV-style line plot.

Uses METHOD_COLORS + METHOD_MARKERS to match the ECCV CRAN-PM paper.
CRAN-PM is plotted with thick line + filled stars to stand out.

Outputs paper/figures/fig_method_comparison.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import (
    METHOD_COLORS,
    METHOD_MARKERS,
    apply_paper_style,
    load_method_metrics,
    save_figure,
)

LEAD_TIMES = [1, 2, 3]
METHOD_ORDER = ["ConvLSTM", "SimVP", "Earthformer", "ClimaX", "TopoFlow", "CRAN-PM"]


def _draw(ax, metrics, key, ylabel, title, panel: str):
    for m in METHOD_ORDER:
        if m not in metrics:
            continue
        ys = []
        for lt in LEAD_TIMES:
            v = metrics[m].get(f"T+{lt}", {}).get("1km", {}).get(key, np.nan)
            ys.append(v if v is not None else np.nan)
        color = METHOD_COLORS[m]
        marker = METHOD_MARKERS[m]
        is_ours = (m == "CRAN-PM")
        ax.plot(
            LEAD_TIMES, ys,
            color=color,
            marker=marker,
            markersize=(11 if is_ours else 7),
            markerfacecolor=color if is_ours else "white",
            markeredgecolor=color,
            markeredgewidth=1.3,
            linewidth=(2.5 if is_ours else 1.5),
            zorder=(5 if is_ours else 3),
            label=(f"{m} (Ours)" if is_ours else m),
        )
        # Improvement annotation for CRAN-PM at each lead time.
        if is_ours:
            for lt, y in zip(LEAD_TIMES, ys):
                # Find best non-CRAN-PM at this lead time.
                others = []
                for m2 in METHOD_ORDER:
                    if m2 == "CRAN-PM" or m2 not in metrics:
                        continue
                    v2 = metrics[m2].get(f"T+{lt}", {}).get("1km", {}).get(key, np.nan)
                    if np.isfinite(v2):
                        others.append(v2)
                if not others:
                    continue
                if key == "rmse":
                    ref = min(others)
                    if not np.isfinite(ref) or ref == 0:
                        continue
                    delta_pct = 100.0 * (y - ref) / ref
                    txt = f"{delta_pct:+.1f}%"
                else:  # R²: report absolute delta (R² range = [-inf, 1])
                    ref = max(others)
                    if not np.isfinite(ref):
                        continue
                    txt = f"+{y - ref:.2f}" if y >= ref else f"{y - ref:.2f}"
                ax.annotate(
                    txt, xy=(lt, y),
                    xytext=(7, -14 if key == "rmse" else 12),
                    textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold",
                )

    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(LEAD_TIMES)
    ax.set_xticklabels([f"T+{lt}\n({lt*24}h)" for lt in LEAD_TIMES])
    ax.set_title(title, pad=6)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, panel, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold")
    if key == "r2":
        ax.axhline(0, color="#444", linewidth=0.5)


def main() -> None:
    apply_paper_style()
    metrics = load_method_metrics()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    _draw(axes[0], metrics, "rmse", "RMSE (µg m$^{-3}$)",
          "Evaluated at 1 km", "(a)")
    _draw(axes[1], metrics, "r2", "R$^2$",
          "Evaluated at 1 km", "(b)")

    # Shared legend below.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    fig.suptitle("Forecast skill on the European 2022 test set",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_figure(fig, "fig_method_comparison")
    plt.close(fig)
    print("Saved: fig_method_comparison.{pdf,png}")


if __name__ == "__main__":
    main()
