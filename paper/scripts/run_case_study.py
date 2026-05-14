"""Run a CRAN-PM forecast over a date window and save predictions for a case study.

Used by sbatch_case_study.sh. Builds ForecastInputs from the global zarrs
via :class:`cranpm.inference.europe_inputs.EuropeInputsBuilder`, runs
:meth:`CRANPMForecaster.predict` once per day, and writes a zarr with the
keys consumed by ``fig_case_study.py``::

    time        : (T,)         array of date strings (YYYY-MM-DD)
    gt_t1       : (T, H, W)    GHAP analysis at t+1 (the verification target)
    cranpm_t1   : (T, H, W)    CRAN-PM T+1 forecast
    cams_t1     : (T, H, W)    CAMS analysis at t+1 (resampled to GHAP grid)
    persistence_t1 : (T, H, W) GHAP analysis at t (persistence baseline)
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import zarr


def daterange(start: str, end: str):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    while s <= e:
        yield s
        s += dt.timedelta(days=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path or HF id of the CRAN-PM checkpoint.")
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--lead-times", nargs="+", type=int, default=[1])
    parser.add_argument("--output-zarr", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args(argv)

    from cranpm import CRANPMForecaster
    from cranpm.inference.europe_inputs import EuropeInputsBuilder
    from cranpm.inference.tiling import GHAP_H, GHAP_W

    fc = CRANPMForecaster.from_pretrained(
        args.checkpoint, device=args.device, precision=args.precision,
    )
    builder = EuropeInputsBuilder()

    dates = list(daterange(args.date_start, args.date_end))
    n = len(dates)
    print(f"Running CRAN-PM on {n} days: {dates[0]} -> {dates[-1]}")

    out_path = Path(args.output_zarr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out_path), mode="w")

    time_arr = root.create_dataset(
        "time", shape=(n,), dtype="<U10",
    )
    cranpm = root.create_dataset(
        "cranpm_t1", shape=(n, GHAP_H, GHAP_W), dtype="float32",
        chunks=(1, 512, 512),
    )
    gt = root.create_dataset(
        "gt_t1", shape=(n, GHAP_H, GHAP_W), dtype="float32",
        chunks=(1, 512, 512),
    )
    persist = root.create_dataset(
        "persistence_t1", shape=(n, GHAP_H, GHAP_W), dtype="float32",
        chunks=(1, 512, 512),
    )
    cams = root.create_dataset(
        "cams_t1", shape=(n, GHAP_H, GHAP_W), dtype="float32",
        chunks=(1, 512, 512),
    )

    for i, d in enumerate(dates):
        doy = d.timetuple().tm_yday - 1  # 0-based
        year = d.year
        try:
            inp_today = builder.build(year=year, day_idx=doy)
        except Exception as exc:
            print(f"[{d}] input build failed: {exc}")
            continue

        # Persistence baseline = today's GHAP carried forward.
        persist[i] = inp_today.ghap_t0

        # Verification ground truth at t+1 = day + 1 GHAP.
        try:
            inp_next = builder.build(year=year, day_idx=min(doy + 1, 364))
            gt[i] = inp_next.ghap_t0
        except Exception:
            gt[i] = inp_today.ghap_t0  # fall back to persistence

        # CRAN-PM T+1 forecast.
        try:
            pred = fc.predict(inp_today, lead_time=1, batch_size=2)
        except Exception as exc:
            print(f"[{d}] predict failed: {exc}")
            continue
        cranpm[i] = pred

        # CAMS at t+1, resampled from 0.4 deg to the GHAP grid (nearest).
        try:
            from cranpm.inference.europe_inputs import EuropeInputsBuilder as _EIB
            cams_today = _EIB()._read_cams_day(year, min(doy + 1, 364))[0]
            y_idx = np.linspace(0, cams_today.shape[0] - 1, GHAP_H).round().astype(int)
            x_idx = np.linspace(0, cams_today.shape[1] - 1, GHAP_W).round().astype(int)
            # CAMS units kg/m^3 -> ug/m^3 (factor 1e9).
            cams[i] = (cams_today[y_idx[:, None], x_idx[None, :]] * 1.0e9).astype(np.float32)
        except Exception as exc:
            print(f"[{d}] cams resample failed: {exc}")
            cams[i] = np.zeros((GHAP_H, GHAP_W), dtype=np.float32)

        time_arr[i] = d.isoformat()
        print(f"  [{i+1:3d}/{n}] {d}: pred range [{pred.min():.1f}, {pred.max():.1f}]")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
