"""Tests for cranpm.config path resolution and env-var overrides."""

import os
from pathlib import Path

from cranpm import config


def test_default_paths_under_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CRANPM_DATA_ROOT", raising=False)
    monkeypatch.delenv("CRANPM_CACHE_DIR", raising=False)
    paths = config.get_paths()
    assert paths.data_root == tmp_path / ".cranpm" / "data"
    assert paths.cache_dir == tmp_path / ".cranpm" / "cache"
    assert paths.cache_dir.exists()


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CRANPM_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CRANPM_CACHE_DIR", str(tmp_path / "cache"))
    paths = config.get_paths()
    assert paths.data_root == tmp_path / "data"
    assert paths.cache_dir == tmp_path / "cache"


def test_derived_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("CRANPM_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("CRANPM_CACHE_DIR", str(tmp_path / "c"))
    paths = config.get_paths()
    assert paths.era5_dir == tmp_path / "era5_europe_daily"
    assert paths.cams_dir == tmp_path / "cams_europe"
    assert paths.ghap_dir == tmp_path / "ghap_pm25_europe_daily"
    assert paths.checkpoints_dir == tmp_path / "c" / "checkpoints"


def test_hf_token_fallback(monkeypatch):
    monkeypatch.delenv("CRANPM_HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "abc123")
    assert config.get_hf_token() == "abc123"

    monkeypatch.setenv("CRANPM_HF_TOKEN", "xyz")
    assert config.get_hf_token() == "xyz"
