"""Normalization constants used at inference time.

These values mirror the ones used during training (see
:mod:`cranpm.data.europe_dataset`). They are duplicated here so that a
caller doing inference need not import the heavy data pipeline (zarr,
pytorch-lightning) just to know how to normalise an input array.
"""

from __future__ import annotations

import numpy as np

GHAP_MEAN = 15.0
GHAP_STD = 20.0
ELEV_MEAN = 300.0
ELEV_STD = 500.0

# ERA5 (5 surface + 25 pressure-level channels). Order:
#   u10, v10, t2m, msl, sp,
#   then for each variable in [t, u, v, q, z]:
#       at pressure levels 1000, 925, 850, 700, 500 hPa.
ERA5_CHANNEL_ORDER = (
    "u10", "v10", "t2m", "msl", "sp",
    "t_1000", "t_925", "t_850", "t_700", "t_500",
    "u_1000", "u_925", "u_850", "u_700", "u_500",
    "v_1000", "v_925", "v_850", "v_700", "v_500",
    "q_1000", "q_925", "q_850", "q_700", "q_500",
    "z_1000", "z_925", "z_850", "z_700", "z_500",
)
N_ERA5 = len(ERA5_CHANNEL_ORDER)
assert N_ERA5 == 30

ERA5_MEANS = np.array(
    [0.0, 0.0, 280.0, 101325.0, 97000.0]
    + [260.0] * 5
    + [0.0] * 5
    + [0.0] * 5
    + [0.003] * 5
    + [50000.0] * 5,
    dtype=np.float32,
)
ERA5_STDS = np.array(
    [10.0, 10.0, 20.0, 1500.0, 5000.0]
    + [30.0] * 5
    + [15.0] * 5
    + [10.0] * 5
    + [0.004] * 5
    + [40000.0] * 5,
    dtype=np.float32,
)

# CAMS (5 pollutants). Order matches the training pipeline.
CAMS_CHANNEL_ORDER = ("no2", "o3", "so2", "co", "pm10")
CAMS_MEANS = np.array([2.15, 71.75, 0.85, 140.8, 14.0], dtype=np.float32)
CAMS_STDS = np.array([3.52, 18.18, 1.69, 36.7, 13.7], dtype=np.float32)
N_CAMS = len(CAMS_CHANNEL_ORDER)
