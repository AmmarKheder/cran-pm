"""CRAN-PM: Cross-Resolution Attention Network for high-resolution PM2.5 forecasting."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cranpm")
except PackageNotFoundError:
    __version__ = "0.1.0+local"

from .inference.forecaster import CRANPMForecaster, ForecastInputs
from .models.multiscale_topoflow import MultiScaleTopoFlow
from .models.topoflow_block import TopoFlowAttention, TopoFlowBlock
from .models.wind_scan import RegionalWindScanner, WindScanner
from .models.global_branch import GlobalBranch
from .models.local_branch import LocalBranch
from .models.cross_attention import CrossAttentionBridge
from .models.prediction_head import PredictionHead, CNNDecoder
from .training.lightning_module import MultiScaleTopoFlowLightning
from .data.europe_dataset import EuropeDataModule, EuropeMultiScaleDataset

__all__ = [
    "__version__",
    # Public forecasting API
    "CRANPMForecaster",
    "ForecastInputs",
    # Core building blocks
    "MultiScaleTopoFlow",
    "MultiScaleTopoFlowLightning",
    "EuropeMultiScaleDataset",
    "EuropeDataModule",
    "GlobalBranch",
    "LocalBranch",
    "CrossAttentionBridge",
    "PredictionHead",
    "CNNDecoder",
    "TopoFlowAttention",
    "TopoFlowBlock",
    "WindScanner",
    "RegionalWindScanner",
]
