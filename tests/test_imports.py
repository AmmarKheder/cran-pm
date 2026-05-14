"""Smoke tests: every public import must succeed without optional extras."""

import importlib

import pytest

PUBLIC_MODULES = [
    "cranpm",
    "cranpm.config",
    "cranpm.models",
    "cranpm.models.multiscale_topoflow",
    "cranpm.models.global_branch",
    "cranpm.models.local_branch",
    "cranpm.models.cross_attention",
    "cranpm.models.topoflow_block",
    "cranpm.models.wind_scan",
    "cranpm.models.prediction_head",
    "cranpm.data",
    "cranpm.data.europe_dataset",
    "cranpm.data.patch_extractor",
    "cranpm.training",
    "cranpm.training.lightning_module",
    "cranpm.training.loss",
    "cranpm.utils",
    "cranpm.utils.pos_embed",
    "cranpm.inference",
    "cranpm.gpu",
    "cranpm.cli.main",
]


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_version_exposed() -> None:
    import cranpm

    assert isinstance(cranpm.__version__, str)
    assert len(cranpm.__version__) > 0


def test_main_classes_importable() -> None:
    from cranpm import (
        CrossAttentionBridge,
        EuropeDataModule,
        EuropeMultiScaleDataset,
        GlobalBranch,
        LocalBranch,
        MultiScaleTopoFlow,
        MultiScaleTopoFlowLightning,
        PredictionHead,
    )

    assert all(
        cls is not None
        for cls in (
            MultiScaleTopoFlow,
            MultiScaleTopoFlowLightning,
            EuropeDataModule,
            EuropeMultiScaleDataset,
            GlobalBranch,
            LocalBranch,
            CrossAttentionBridge,
            PredictionHead,
        )
    )
