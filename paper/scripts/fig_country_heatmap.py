"""Figure: per-country MAE heatmap (4 methods x 8 EU countries).

Source: country_mae_cache.npz from evaluation_2022/.

Outputs paper/figures/fig_country_heatmap.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import apply_paper_style, load_country_mae, save_figure

METHOD_ORDER = ["CAMS", "ClimaX", "TopoFlow", "CRAN-PM"]


def main() -> None:
    apply_paper_style()

    mae = load_country_mae()
    methods = [m for m in METHOD_ORDER if m in mae]
    countries = list(mae[methods[0]].keys())

    matrix = np.array([[mae[m][c] for c in countries] for m in methods], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    im = ax.imshow(
        matrix, cmap="YlOrRd", aspect="auto",
        vmin=0, vmax=max(8.0, float(np.nanmax(matrix))),
    )

    ax.set_xticks(np.arange(len(countries)))
    ax.set_xticklabels(countries, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Country")
    ax.set_title("Per-country MAE (µg m$^{-3}$), T+1, EEA stations 2022")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v > 5.0 else "#222"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("MAE (µg m$^{-3}$)")
    cbar.outline.set_linewidth(0.4)

    fig.tight_layout()
    save_figure(fig, "fig_country_heatmap")
    plt.close(fig)
    print("Saved: fig_country_heatmap.{pdf,png}")


if __name__ == "__main__":
    main()
