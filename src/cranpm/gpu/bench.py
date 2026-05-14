"""Reproducible single-GPU inference benchmark for CRAN-PM.

Measures throughput (maps/s), latency (ms/tile) and peak memory for one
or more precisions on the configured device. Writes JSON suitable for the
paper figure :py:mod:`paper.scripts.fig_gpu_benchmarks`.

Usage::

    python -m cranpm.gpu.bench \\
        --backend rocm \\
        --checkpoint /path/to/topoflow-016.ckpt \\
        --precision bf16 fp16 fp32 \\
        --batch-sizes 1 2 4 8 \\
        --warmup-iter 5 --measure-iter 50 \\
        --output bench_rocm.json

The harness uses synthetic inputs of the correct shape (no real data
required) so that hardware comparisons are not confounded by I/O.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


PRECISION_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _gpu_info(device: torch.device) -> dict:
    info = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(device)
        info["device_capability"] = torch.cuda.get_device_capability(device)
        # Heuristic: hip vs cuda
        info["torch_hip"] = bool(getattr(torch.version, "hip", None))
        info["torch_cuda"] = torch.version.cuda or ""
    return info


def _make_synthetic_batch(batch_size: int, device: torch.device,
                           dtype: torch.dtype, model_cfg: dict) -> dict:
    """Generate one model-sized batch with the right shapes."""
    img_g = tuple(model_cfg["global_img_size"])
    img_l = tuple(model_cfg["local_img_size"])
    return dict(
        era5=torch.randn(batch_size, model_cfg["era5_channels"], *img_g,
                         device=device, dtype=dtype),
        elevation_coarse=torch.randn(batch_size, *img_g,
                                      device=device, dtype=dtype),
        ghap_patch=torch.randn(batch_size, model_cfg["local_channels"], *img_l,
                                device=device, dtype=dtype),
        elevation_hires=torch.randn(batch_size, *img_l,
                                     device=device, dtype=dtype),
        lead_time=torch.ones(batch_size, device=device, dtype=dtype),
        patch_center=torch.zeros(batch_size, 2, device=device, dtype=dtype),
        wind_at_patch=None,
    )


def benchmark_once(model, batch_size: int, precision: str,
                   warmup_iter: int, measure_iter: int,
                   device: torch.device, model_cfg: dict) -> dict:
    dtype = PRECISION_DTYPES[precision]
    model_dtype = dtype if precision == "fp32" else torch.float32  # autocast handles low precision
    model = model.to(model_dtype)

    # Build batch in fp32; autocast wraps the forward.
    batch = _make_synthetic_batch(batch_size, device, torch.float32, model_cfg)

    autocast = torch.autocast(
        device_type=device.type, dtype=dtype, enabled=precision != "fp32",
    )

    # Warm-up.
    with torch.inference_mode(), autocast:
        for _ in range(warmup_iter):
            _ = model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Measure.
    t0 = time.perf_counter()
    with torch.inference_mode(), autocast:
        for _ in range(measure_iter):
            _ = model(**batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    per_call_ms = 1000.0 * elapsed / measure_iter
    tiles_per_sec = measure_iter * batch_size / elapsed
    # A full European map is 126 overlapping tiles in our standard layout.
    maps_per_sec = tiles_per_sec / 126.0
    peak_mem = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda" else 0.0
    )
    return {
        "batch_size": batch_size,
        "precision": precision,
        "per_call_ms": per_call_ms,
        "tiles_per_sec": tiles_per_sec,
        "maps_per_sec": maps_per_sec,
        "peak_mem_mib": peak_mem,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CRAN-PM single-GPU bench")
    parser.add_argument("--backend", choices=["rocm", "cuda", "cpu"],
                        default="rocm", help="Reported in JSON; affects device detection.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a CRAN-PM checkpoint (HF dir or .ckpt).")
    parser.add_argument("--precision", nargs="+",
                        default=["bf16", "fp32"],
                        choices=list(PRECISION_DTYPES))
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 2, 4, 8])
    parser.add_argument("--warmup-iter", type=int, default=5)
    parser.add_argument("--measure-iter", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.backend == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    print(f"=== CRAN-PM GPU benchmark on {device} ===")
    print(_gpu_info(device))

    from cranpm.inference.checkpoint import Checkpoint
    from cranpm.inference.forecaster import _filter_model_kwargs
    from cranpm.models.multiscale_topoflow import MultiScaleTopoFlow

    ckpt = Checkpoint.from_path(Path(args.checkpoint))
    model_cfg = ckpt.config.get("model", ckpt.config)
    if isinstance(model_cfg.get("global_img_size"), list):
        model_cfg = dict(model_cfg)
        model_cfg["global_img_size"] = tuple(model_cfg["global_img_size"])
        model_cfg["local_img_size"] = tuple(model_cfg["local_img_size"])
    model = MultiScaleTopoFlow(**_filter_model_kwargs(model_cfg))
    missing, unexpected = model.load_state_dict(ckpt.state_dict, strict=False)
    if missing:
        print(f"warning: {len(missing)} missing keys")
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected keys")
    model = model.eval().to(device)

    results = []
    for prec in args.precision:
        for bs in args.batch_sizes:
            print(f"\n-- precision={prec}, batch={bs} --")
            try:
                r = benchmark_once(
                    model, bs, prec,
                    warmup_iter=args.warmup_iter,
                    measure_iter=args.measure_iter,
                    device=device, model_cfg=model_cfg,
                )
                print(f"  {r['per_call_ms']:.2f} ms/call, "
                      f"{r['tiles_per_sec']:.1f} tiles/s, "
                      f"{r['maps_per_sec']:.3f} maps/s, "
                      f"{r['peak_mem_mib']:.0f} MiB peak")
                results.append(r)
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                results.append({
                    "batch_size": bs, "precision": prec,
                    "error": str(e)[:200],
                })

    out = {
        "backend": args.backend,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_hip": bool(getattr(torch.version, "hip", None)),
        "torch_cuda": torch.version.cuda or "",
        "checkpoint": args.checkpoint,
        "warmup_iter": args.warmup_iter,
        "measure_iter": args.measure_iter,
        "device_info": _gpu_info(device),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
