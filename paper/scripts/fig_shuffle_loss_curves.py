#!/usr/bin/env python3
"""Plot the train/val loss curves of the 3 shuffle-ablation training runs.

Reads TensorBoard event files from
  topoflow_europe/logs/multiscale_topoflow/version_{38,39,40}/
(version_38 = wind, version_39 = random, version_40 = raster, confirmed
from hparams.yaml in each run).

Outputs:
  paper/figures/fig_shuffle_loss_curves.{pdf,png}
  paper/figures/shuffle_loss_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


VERSIONS = {
    "wind":   38,
    "random": 39,
    "raster": 40,
}

TB_ROOT = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe/"
                "logs/multiscale_topoflow")

# Order in which the 3 panels are stacked / curves are drawn.
ORDER = ["wind", "random", "raster"]
COLOURS = {"wind": "#2ca02c", "random": "#d62728", "raster": "#ff7f0e"}


def read_tb(version_dir: Path) -> dict[str, list[tuple[float, float, int]]]:
    ea = EventAccumulator(str(version_dir),
                          size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for tag in ea.Tags()["scalars"]:
        for ev in ea.Scalars(tag):
            out[tag].append((ev.wall_time, ev.value, ev.step))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("/scratch/project_462001140/ammar/eccv/"
                                     "cran-pm/paper/figures/"
                                     "fig_shuffle_loss_curves"))
    args = parser.parse_args()

    runs: dict[str, dict[str, list[tuple[float, float, int]]]] = {}
    for mode, vnum in VERSIONS.items():
        d = TB_ROOT / f"version_{vnum}"
        if not d.exists():
            print(f"[skip] {mode}: {d} missing")
            continue
        runs[mode] = read_tb(d)
        print(f"[{mode} v{vnum}] tags: {list(runs[mode].keys())}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8),
                              gridspec_kw={"wspace": 0.26})
    fig.patch.set_facecolor("white")

    plot_tags = [
        ("train/loss",     "Train loss (per step)", "log"),
        ("val/loss",       "Val loss (per epoch)",  "linear"),
        ("val/rmse",       r"Val RMSE ($\mu$g m$^{-3}$, per epoch)", "linear"),
    ]
    summary = {m: {} for m in ORDER}
    for k, (tag, title, yscale) in enumerate(plot_tags):
        ax = axes[k]
        for mode in ORDER:
            if mode not in runs or tag not in runs[mode]:
                continue
            evs = sorted(runs[mode][tag], key=lambda x: x[2])
            steps = np.array([e[2] for e in evs])
            vals = np.array([e[1] for e in evs])
            ax.plot(steps, vals, lw=1.5, color=COLOURS[mode],
                    label=f"{mode}", marker="o" if "val" in tag else None,
                    markersize=4 if "val" in tag else None)
            if "val/rmse" in tag and len(vals):
                summary[mode]["val_rmse_final"] = float(vals[-1])
                summary[mode]["val_rmse_min"] = float(np.min(vals))
            if "val/loss" in tag and len(vals):
                summary[mode]["val_loss_final"] = float(vals[-1])
            if "train/loss" in tag and len(vals):
                summary[mode]["train_loss_final"] = float(vals[-1])
        ax.set_yscale(yscale)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("global step" if "train" in tag else "epoch",
                       fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=9)
        if k == 0:
            ax.legend(fontsize=10, loc="upper right", framealpha=0.92,
                       title="shuffle mode")
        else:
            ax.legend(fontsize=10, loc="best", framealpha=0.92)

    fig.suptitle(
        "Shuffling ablation — from-scratch training of three "
        "patch-ordering strategies (3 epochs on Europe 2018)",
        fontsize=12.5, y=1.02,
    )

    fig.savefig(str(args.out.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(args.out.with_suffix(".png")), bbox_inches="tight",
                 dpi=180)
    print(f"\nWrote {args.out.with_suffix('.pdf')} and .png")

    with open(str(args.out.parent / "shuffle_loss_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
