"""Figure: software stack / data-flow diagram for the CRAN-PM tool.

Pure matplotlib block diagram (no scientific data needed).
This is the "what is the tool" figure for the GMD paper Section 4.

Outputs paper/figures/fig_software_stack.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from cranpm_plots import apply_paper_style, save_figure


def block(ax, x, y, w, h, text, fc, ec="#333333", fontsize=8.5, text_color="#111"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.8, facecolor=fc, edgecolor=ec, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, color=text_color, zorder=3,
    )


def arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8, shrinkA=2, shrinkB=2),
        zorder=1,
    )


def main() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    palette = {
        "data": "#cfe2f3",
        "loader": "#fde9c4",
        "model": "#d9ead3",
        "infer": "#f4cccc",
        "out": "#e6d6f5",
        "infra": "#eeeeee",
    }

    # --- Layer 1: external data sources ---
    ax.text(0.1, 7.6, "External data (Copernicus CDS / Zenodo)",
            fontsize=8.5, color="#666", style="italic")
    block(ax, 0.2, 6.6, 1.7, 0.7, "ERA5\n(meteo, 25 km)", palette["data"])
    block(ax, 2.1, 6.6, 1.7, 0.7, "CAMS\n(chemistry)", palette["data"])
    block(ax, 4.0, 6.6, 1.7, 0.7, "GHAP\n(PM$_{2.5}$, 1 km)", palette["data"])
    block(ax, 5.9, 6.6, 1.7, 0.7, "GMTED2010\n(elevation, 250 m)", palette["data"])
    block(ax, 7.8, 6.6, 1.7, 0.7, "EEA stations\n(validation)", palette["data"])

    # --- Layer 2: download / preprocessing ---
    ax.text(0.1, 5.7, "cranpm.cli download / cranpm.data preprocessors",
            fontsize=8.5, color="#666", style="italic")
    block(ax, 0.5, 4.8, 4.0, 0.7, "Zarr cache\n(per-year, chunked)", palette["loader"])
    block(ax, 5.0, 4.8, 4.0, 0.7, "Patch extractor\n(hotspot-weighted, augment)", palette["loader"])

    arrow(ax, 1.0, 6.6, 1.5, 5.5)
    arrow(ax, 2.9, 6.6, 2.5, 5.5)
    arrow(ax, 4.8, 6.6, 3.5, 5.5)
    arrow(ax, 6.7, 6.6, 7.0, 5.5)
    arrow(ax, 8.6, 6.6, 8.0, 5.5)

    # --- Layer 3: model ---
    ax.text(0.1, 3.9, "cranpm.models — multi-scale Vision Transformer (this paper)",
            fontsize=8.5, color="#666", style="italic")
    block(ax, 0.5, 3.0, 2.7, 0.7, "Global branch\n(ViT, 25 km)", palette["model"])
    block(ax, 3.4, 3.0, 2.7, 0.7, "Cross-resolution\nattention bridge", palette["model"])
    block(ax, 6.3, 3.0, 2.7, 0.7, "Local branch\n(TopoFlow, 1 km)", palette["model"])
    block(ax, 9.2, 3.0, 2.4, 0.7, "Δ-decoder\n(zero-init)", palette["model"])
    arrow(ax, 3.2, 3.35, 3.4, 3.35)
    arrow(ax, 6.1, 3.35, 6.3, 3.35)
    arrow(ax, 9.0, 3.35, 9.2, 3.35)

    arrow(ax, 4.5, 4.8, 4.5, 3.7)

    # --- Layer 4: training + inference ---
    ax.text(0.1, 2.1, "cranpm.training (Lightning) | cranpm.inference (public API) | cranpm.gpu (CUDA + ROCm)",
            fontsize=8.5, color="#666", style="italic")
    block(ax, 0.5, 1.2, 3.4, 0.7, "Training (Lightning DDP)", palette["infer"])
    block(ax, 4.1, 1.2, 3.4, 0.7, "Inference (single GPU, bf16)", palette["infer"])
    block(ax, 7.7, 1.2, 3.9, 0.7, "Benchmarks (CUDA / ROCm)", palette["infer"])
    arrow(ax, 5.5, 3.0, 2.2, 1.9)
    arrow(ax, 5.5, 3.0, 5.8, 1.9)

    # --- Layer 5: outputs ---
    ax.text(0.1, 0.3, "Outputs",
            fontsize=8.5, color="#666", style="italic")
    block(ax, 0.5, -0.6, 2.6, 0.7, "Forecast NetCDF\n(1 km × 1-4 days)", palette["out"])
    block(ax, 3.3, -0.6, 2.6, 0.7, "HuggingFace\nmodel weights", palette["out"])
    block(ax, 6.1, -0.6, 2.6, 0.7, "Zenodo DOI\n(code + data)", palette["out"])
    block(ax, 8.9, -0.6, 2.7, 0.7, "Gradio Space\n(public demo)", palette["out"])
    arrow(ax, 5.8, 1.2, 1.8, 0.1)
    arrow(ax, 5.8, 1.2, 7.4, 0.1)
    arrow(ax, 5.8, 1.2, 10.2, 0.1)

    # adjust ylim for outputs
    ax.set_ylim(-1.0, 8.0)

    fig.suptitle("CRAN-PM v1.0 — software stack and data flow", y=0.97, fontsize=11)
    save_figure(fig, "fig_software_stack")
    plt.close(fig)
    print("Saved: fig_software_stack.{pdf,png}")


if __name__ == "__main__":
    main()
