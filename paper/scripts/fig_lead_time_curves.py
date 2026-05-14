"""Figure: forecast skill vs lead time per method (RMSE + R^2).

Two side-by-side line plots, each method colored from METHOD_COLORS.
Source: main_table_*.json from evaluation_2022/.

Outputs paper/figures/fig_lead_time_curves.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import (
    METHOD_COLORS,
    apply_paper_style,
    load_method_metrics,
    save_figure,
)

LEAD_TIMES = [1, 2, 3]
METHOD_ORDER = [
    "Persistence", "CAMS", "ConvLSTM", "SimVP",
    "Earthformer", "ClimaX", "TopoFlow", "CRAN-PM",
]


def main() -> None:
    apply_paper_style()
    metrics = load_method_metrics()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)

    for ax, metric, ylabel in zip(
        axes, ["rmse", "r2"], ["RMSE (µg m$^{-3}$)", "R$^2$"]
    ):
        for m in METHOD_ORDER:
            if m not in metrics:
                continue
            vals = []
            for lt in LEAD_TIMES:
                v = metrics[m].get(f"T+{lt}", {}).get("1km", {}).get(metric, np.nan)
                vals.append(v if v is not None else np.nan)
            color = METHOD_COLORS.get(m, "#999999")
            lw = 2.2 if m == "CRAN-PM" else 1.2
            marker = "o" if m == "CRAN-PM" else "."
            ms = 6 if m == "CRAN-PM" else 4
            ax.plot(LEAD_TIMES, vals, color=color, marker=marker,
                    markersize=ms, linewidth=lw, label=m)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(LEAD_TIMES)
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        if metric == "r2":
            ax.axhline(0, color="#444", linewidth=0.5)

    axes[0].legend(loc="best", ncol=2, fontsize=7.5)
    fig.suptitle("Forecast skill degradation with lead time (Europe, 2022)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "fig_lead_time_curves")
    plt.close(fig)
    print("Saved: fig_lead_time_curves.{pdf,png}")


if __name__ == "__main__":
    main()
