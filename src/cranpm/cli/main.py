"""Entry point for the ``cranpm`` console script.

The full CLI surface (forecast / train / download / benchmark) is wired
up in phase 2. This stub keeps the entry point importable so packaging
and ``cranpm --version`` work today.
"""

from __future__ import annotations

import argparse
import sys

from cranpm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cranpm",
        description="CRAN-PM: high-resolution PM2.5 forecasting toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"cranpm {__version__}")
    sub = parser.add_subparsers(dest="command")
    fc = sub.add_parser("forecast", help="Run a pan-European PM2.5 forecast.")
    fc.add_argument("--checkpoint", required=True,
                    help="HF Hub repo id or local path to a checkpoint directory / .ckpt file.")
    fc.add_argument("--inputs", required=True,
                    help="Path to a directory containing era5.npy, cams.npy, ghap_t0.npy, "
                         "ghap_tm1.npy, elev_coarse.npy, elev_hires.npy.")
    fc.add_argument("--lead-time", type=int, default=1)
    fc.add_argument("--output", required=True, help="Output .npy file for the forecast.")
    fc.add_argument("--device", default=None)
    fc.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])

    sub.add_parser("train", help="Train or fine-tune a model (phase 2).")
    sub.add_parser("benchmark", help="Run GPU benchmarks (phase 3).")
    sub.add_parser("download", help="Download input data from CDS (phase 2).")
    return parser


def _cmd_forecast(args) -> int:
    from pathlib import Path

    import numpy as np

    from cranpm import CRANPMForecaster, ForecastInputs

    inp_dir = Path(args.inputs)
    inputs = ForecastInputs(
        era5_global=np.load(inp_dir / "era5.npy"),
        elev_coarse=np.load(inp_dir / "elev_coarse.npy"),
        ghap_t0=np.load(inp_dir / "ghap_t0.npy"),
        ghap_tm1=np.load(inp_dir / "ghap_tm1.npy"),
        elev_hires=np.load(inp_dir / "elev_hires.npy"),
    )
    fc = CRANPMForecaster.from_pretrained(
        args.checkpoint, device=args.device, precision=args.precision,
    )
    forecast = fc.predict(inputs, lead_time=args.lead_time, verbose=True)
    np.save(args.output, forecast)
    print(f"Wrote forecast {forecast.shape} to {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "forecast":
        return _cmd_forecast(args)
    print(f"`cranpm {args.command}` is not implemented yet (planned for phase 2/3).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
