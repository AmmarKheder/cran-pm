"""Figure: ablation bar chart — RMSE / R^2 delta vs full model per component.

Source: ablation_table_full.json from evaluation_2022/.

Outputs paper/figures/fig_ablation_bars.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np

from cranpm_plots import apply_paper_style, save_figure

EVAL = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/evaluation_2022")


def main() -> None:
    apply_paper_style()
    with open(EVAL / "ablation_table_full.json") as fh:
        data = json.load(fh)

    # Take the variants under part_a, plus the "(vii) Fine queries Coarse (ours)" as anchor.
    variants = data.get("variants_part_a", {})
    if not variants:
        print("No variants_part_a found, aborting.")
        return

    # Order: ours first (or last), others sorted by RMSE
    ours_key = next((k for k in variants if "ours" in k.lower() or "fine queries coarse" in k.lower()), None)
    other_keys = [k for k in variants if k != ours_key]
    other_keys = sorted(other_keys, key=lambda k: variants[k]["rmse"])
    order = ([ours_key] if ours_key else []) + other_keys

    labels = [k.replace("(", "").replace(")", "").strip() for k in order]
    rmse = [variants[k]["rmse"] for k in order]
    r2 = [variants[k]["r2"] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    colors = ["#1f77b4" if (ours_key and k == ours_key) else "#888888" for k in order]

    for ax, vals, ylabel, fmt in zip(
        axes, [rmse, r2], ["RMSE (µg m$^{-3}$)", "R$^2$"], ["%.2f", "%.3f"]
    ):
        bars = ax.bar(range(len(labels)), vals, color=colors,
                      edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.05 if max(vals) > 1 else 0.01),
                    fmt % v, ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        "Ablation: cross-attention variants (Europe 2022 test, T+1, 90 days)",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig_ablation_bars")
    plt.close(fig)
    print("Saved: fig_ablation_bars.{pdf,png}")


if __name__ == "__main__":
    main()
