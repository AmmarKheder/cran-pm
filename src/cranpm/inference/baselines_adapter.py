"""Adaptors that expose ConvLSTM / SimVP / Earthformer / ClimaX baselines
behind the same `.predict(inputs, lead_time)` interface as
:class:`CRANPMForecaster`.

This lets the physics-ablation runner subject every ML baseline to the
same battery of test-time interventions, so the GMD paper can show
"how physics-aware is each model" rather than just "which model has
the lowest RMSE".

Each baseline lives in the original `topoflow_europe.baselines` package
(not migrated to cranpm.* yet because they are not part of the
distributed model). We import them lazily so the rest of cranpm stays
torch-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from cranpm.config import get_default_device
from cranpm.inference.forecaster import ForecastInputs
from cranpm.inference.normalize import GHAP_MEAN, GHAP_STD

# We must add the legacy research repo to sys.path to import baselines.
_LEGACY_REPO = Path("/scratch/project_462001140/ammar/eccv/topoflow_europe")
if _LEGACY_REPO.exists() and str(_LEGACY_REPO) not in sys.path:
    sys.path.insert(0, str(_LEGACY_REPO))


class BaselineForecaster:
    """Uniform `.predict(inputs, lead_time)` for ConvLSTM / SimVP / Earthformer / ClimaX.

    The baselines were trained with the same input layout as CRAN-PM but
    without the cross-attention bridge, so the same ablations apply.
    """

    def __init__(self, name: str, model: torch.nn.Module, device: str,
                 precision: str = "fp32"):
        self.name = name
        self.model = model.eval().to(device)
        self.device = torch.device(device)
        self.precision = precision

    @classmethod
    def from_checkpoint(cls, name: str, checkpoint: Path,
                         device: str | None = None,
                         precision: str = "fp32") -> "BaselineForecaster":
        device = device or get_default_device()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {k.removeprefix("model."): v for k, v in sd.items()}

        if name == "convlstm":
            from baselines.convlstm_baseline import ConvLSTMBaseline as Cls
        elif name == "simvp":
            from baselines.simvp_baseline import SimVPBaseline as Cls
        elif name == "earthformer":
            from baselines.earthformer_baseline import EarthformerBaseline as Cls
        elif name == "climax":
            from baselines.climax_baseline import ClimaXBaseline as Cls
        else:
            raise ValueError(f"unknown baseline: {name}")

        cfg = ck.get("hyper_parameters", {}).get("config", {})
        model = Cls(**cfg.get("model", cfg)) if "model" in cfg else Cls()
        model.load_state_dict(sd, strict=False)
        return cls(name=name, model=model, device=device, precision=precision)

    @torch.no_grad()
    def predict(self, inputs: ForecastInputs, lead_time: int = 1) -> np.ndarray:
        """Run a baseline forecast over the full Europe domain.

        We re-use the CRAN-PM tile loop but swap the model's forward call.
        Baselines that don't accept patch_center / wind_at_patch ignore them.
        """
        from cranpm.inference.forecaster import CRANPMForecaster
        from cranpm.inference.tiling import iter_tiles, stitch_tiles

        # Normalisation is identical across baselines (they were trained on
        # the same EuropeMultiScaleDataset).
        fc_helper = object.__new__(CRANPMForecaster)
        fc_helper.device = self.device
        fc_helper.precision = self.precision
        fc_helper._dtype_map = {
            "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
        }

        era5 = fc_helper._normalise_era5(inputs.era5_global)
        elev_c = (inputs.elev_coarse - 300.0) / 500.0
        ghap_n = (inputs.ghap_t0 - GHAP_MEAN) / GHAP_STD
        ghap_pm1_n = (inputs.ghap_tm1 - GHAP_MEAN) / GHAP_STD
        elev_h_n = (inputs.elev_hires - 300.0) / 500.0

        era5_t = torch.from_numpy(era5).unsqueeze(0).to(self.device)
        elev_c_t = torch.from_numpy(elev_c).unsqueeze(0).to(self.device)

        results = []
        for tile in iter_tiles():
            local = fc_helper._build_local_patch(
                tile, ghap_n, ghap_pm1_n, elev_h_n,
                lats_grid=np.linspace(72.0, 30.08, 4192, dtype=np.float32),
                lons_grid=np.linspace(-25.0, 44.92, 6992, dtype=np.float32),
            )
            local_t = torch.from_numpy(local).unsqueeze(0).to(self.device)
            elev_h_t = torch.from_numpy(
                elev_h_n[tile.row:tile.row + tile.size,
                         tile.col:tile.col + tile.size]
            ).unsqueeze(0).to(self.device)
            lt_t = torch.tensor([float(lead_time)], device=self.device)

            # Baselines have heterogeneous forward signatures. We call by
            # keyword whenever supported, falling back to positional.
            try:
                out = self.model(
                    era5=era5_t, elevation_coarse=elev_c_t,
                    ghap_patch=local_t, elevation_hires=elev_h_t,
                    lead_time=lt_t,
                )
            except TypeError:
                out = self.model(era5_t, local_t, lt_t)
            pred_norm = out[:, 0].float().cpu().numpy()[0]
            pred = pred_norm * GHAP_STD + GHAP_MEAN
            results.append((tile, pred))
        return stitch_tiles(results)
