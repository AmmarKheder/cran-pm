"""Stage ready-to-run CRAN-PM input bundles for the HF Space.

Uses the *tested* EuropeInputsBuilder + CRANPMForecaster internals so the
ERA5/CAMS/GHAP cropping and normalisation match production exactly. For
each (date, city) we dump one .npz with the tensors model.forward()
needs, plus the raw ground truth for scoring.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

SRC = "/scratch/project_462001140/ammar/eccv/cran-pm/src"
sys.path.insert(0, SRC)

from cranpm.inference.forecaster import CRANPMForecaster      # noqa: E402
from cranpm.inference.europe_inputs import EuropeInputsBuilder  # noqa: E402
from cranpm.inference.normalize import (                       # noqa: E402
    GHAP_MEAN, GHAP_STD, ELEV_MEAN, ELEV_STD,
)

GHAP_H, GHAP_W = 4192, 6992
LAT_NORTH, LON_WEST, GHAP_RES = 72.0, -25.0, 0.01
OUT = Path("/tmp/cranpm_hf_space/samples")
OUT.mkdir(parents=True, exist_ok=True)

# (label, date_str, doy 0-based, centre lat, centre lon)
SAMPLES = [
    ("Po Valley winter haze",  "2022-01-12", 11, 45.4,  9.2),
    ("Paris basin spring",     "2022-04-15", 104, 48.8,  2.3),
    ("Krakow Silesia coal",    "2022-02-15", 45, 50.1,  19.0),
    ("Benelux industrial",     "2022-03-15", 73, 51.2,   4.4),
    ("Madrid summer",          "2022-07-15", 195, 40.4, -3.7),
    ("London autumn",          "2022-10-15", 287, 51.5, -0.1),
]


def main():
    ckpt = ("/scratch/project_462001140/ammar/eccv/topoflow_europe/"
            "checkpoints_v10f/topoflow-018.ckpt")
    fc = CRANPMForecaster.from_pretrained(ckpt, device="cpu",
                                           precision="fp32")
    builder = EuropeInputsBuilder()

    for label, date_str, doy, clat, clon in SAMPLES:
        inp = builder.build(2022, doy)

        # Normalise exactly as forecaster.predict() does.
        era5 = fc._normalise_era5(inp.era5_global)            # (35,168,280)
        elev_c = (inp.elev_coarse - ELEV_MEAN) / ELEV_STD
        ghap_n = (np.nan_to_num(inp.ghap_t0) - GHAP_MEAN) / GHAP_STD
        ghap_pm1_n = (np.nan_to_num(inp.ghap_tm1) - GHAP_MEAN) / GHAP_STD
        elev_h_n = (inp.elev_hires - ELEV_MEAN) / ELEV_STD

        # Tile window centred on the city.
        cy = int(round((LAT_NORTH - clat) / GHAP_RES))
        cx = int(round((clon - LON_WEST) / GHAP_RES))
        r = max(0, min(GHAP_H - 512, cy - 256))
        c = max(0, min(GHAP_W - 512, cx - 256))

        rows = np.arange(r, r + 512, dtype=np.float32)
        cols = np.arange(c, c + 512, dtype=np.float32)
        lat_g = ((LAT_NORTH - rows * GHAP_RES) - 50.0) / 25.0
        lon_g = ((LON_WEST + cols * GHAP_RES) - 10.0) / 35.0
        ghap_patch = np.stack([
            ghap_n[r:r + 512, c:c + 512],                       # PM2.5 t
            elev_h_n[r:r + 512, c:c + 512],                     # elevation
            np.broadcast_to(lat_g[:, None], (512, 512)),        # lat
            np.broadcast_to(lon_g[None, :], (512, 512)),        # lon
            ghap_pm1_n[r:r + 512, c:c + 512],                   # PM2.5 t-1
        ]).astype(np.float32)

        gt = np.nan_to_num(inp.ghap_t0)[r:r + 512, c:c + 512]
        # ghap_t0 is "today"; the +1 ground truth is forecast target —
        # rebuild from next day for honest scoring.
        inp_next = builder.build(2022, min(doy + 1, 363))
        gt = np.nan_to_num(inp_next.ghap_t0)[r:r + 512, c:c + 512]

        pc = np.array([
            ((LAT_NORTH - (r + 256) * GHAP_RES) - 50.0) / 25.0,
            ((LON_WEST + (c + 256) * GHAP_RES) - 10.0) / 35.0,
        ], dtype=np.float32)

        slug = label.lower().replace(" ", "_")
        np.savez_compressed(
            OUT / f"{slug}.npz",
            era5=era5.astype(np.float32),
            elevation_coarse=elev_c.astype(np.float32),
            ghap_patch=ghap_patch,
            elevation_hires=elev_h_n[r:r + 512, c:c + 512].astype(np.float32),
            lead_time=np.array([1.0], np.float32),
            patch_center=pc,
            gt=gt.astype(np.float32),
            meta=np.frombuffer(
                pickle.dumps(dict(label=label, date=date_str,
                                   lat=clat, lon=clon)), np.uint8),
        )
        print(f"  staged {slug}.npz  era5={era5.shape} "
              f"patch={ghap_patch.shape} gt[max={gt.max():.0f}]",
              flush=True)

    print(f"\nDone -> {OUT}")


if __name__ == "__main__":
    main()
