"""Checkpoint loading for CRAN-PM.

Two formats are supported:

* **PyTorch Lightning ``.ckpt``** as produced by the training pipeline.
  Contains the full state dict, hyperparameters and optimizer state.
* **HuggingFace-style** ``model.safetensors`` + ``config.json`` pair.
  Lighter, portable, and the recommended distribution format.

The :class:`Checkpoint` dataclass abstracts over both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
LIGHTNING_PREFIX = "model."  # MultiScaleTopoFlowLightning wraps model as `self.model`.


@dataclass
class Checkpoint:
    """In-memory representation of a CRAN-PM checkpoint."""

    config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    source: Path

    @classmethod
    def from_path(cls, path: Path) -> "Checkpoint":
        path = Path(path)
        if path.is_dir():
            return cls._load_hf_dir(path)
        if path.suffix == ".ckpt":
            return cls._load_lightning(path)
        if path.suffix == ".safetensors":
            return cls._load_hf_dir(path.parent)
        raise ValueError(f"Unrecognised checkpoint path: {path}")

    @classmethod
    def _load_lightning(cls, path: Path) -> "Checkpoint":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        hparams = ck.get("hyper_parameters", {})
        config = hparams.get("config", hparams)
        sd = ck["state_dict"]
        sd = {
            k[len(LIGHTNING_PREFIX):]: v
            for k, v in sd.items()
            if k.startswith(LIGHTNING_PREFIX)
        }
        sd = _convert_conv2d_patch_embed_to_linear(sd)
        config = _infer_config_from_state_dict(sd, fallback_config=config)
        return cls(config=config, state_dict=sd, source=path)

    @classmethod
    def _load_hf_dir(cls, path: Path) -> "Checkpoint":
        from safetensors.torch import load_file

        cfg_path = path / CONFIG_FILENAME
        weights_path = path / WEIGHTS_FILENAME
        if not cfg_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Expected {CONFIG_FILENAME} and {WEIGHTS_FILENAME} in {path}"
            )
        with open(cfg_path) as fh:
            config = json.load(fh)
        sd = load_file(str(weights_path))
        return cls(config=config, state_dict=sd, source=path)

    def to_safetensors_dir(self, out_dir: Path) -> None:
        """Persist as the HuggingFace-style two-file format."""
        from safetensors.torch import save_file

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict, str(out_dir / WEIGHTS_FILENAME))
        with open(out_dir / CONFIG_FILENAME, "w") as fh:
            json.dump(self.config, fh, indent=2, default=str)


def _convert_conv2d_patch_embed_to_linear(
    sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reshape legacy Conv2d patch-embed weights to the F.unfold + Linear layout.

    The CRAN-PM package replaced ``nn.Conv2d`` with ``F.unfold + nn.Linear`` in
    the patch-embed layers to work around a numerical bug in MIOpen on
    AMD MI250X. Older checkpoints carry the Conv2d weights as a 4-D tensor
    of shape ``(out_dim, in_ch, kH, kW)``; the new model expects a 2-D
    Linear weight of shape ``(out_dim, in_ch * kH * kW)``. This helper
    silently reshapes when needed so old checkpoints load on the new code.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.endswith(".patch_embed.weight") and v.ndim == 4:
            out_dim = v.shape[0]
            out[k] = v.reshape(out_dim, -1).contiguous()
        else:
            out[k] = v
    return out


def _infer_config_from_state_dict(
    sd: dict[str, torch.Tensor],
    fallback_config: dict | None = None,
) -> dict:
    """Best-effort reconstruction of the model config when hparams are missing.

    Older Lightning checkpoints sometimes serialise ``hyper_parameters`` as an
    empty dict. In that case we recover the architecture by inspecting tensor
    shapes in the state dict (number of input channels, embedding dimensions,
    patch sizes).
    """
    fallback_config = fallback_config or {}
    if isinstance(fallback_config.get("model"), dict) and fallback_config["model"]:
        return fallback_config

    cfg: dict[str, object] = {}

    gpe = sd.get("global_branch.patch_embed.weight")
    if gpe is not None:
        if gpe.ndim == 4:
            out_dim, in_ch, kH, kW = gpe.shape
        else:
            out_dim, flat = gpe.shape
            # default 8x8 patch for global
            kH = kW = 8
            in_ch = flat // (kH * kW)
        cfg["global_embed_dim"] = int(out_dim)
        cfg["era5_channels"] = int(in_ch)
        cfg["global_patch_size"] = int(kH)
        cfg["global_img_size"] = (168, 280)

    lpe = sd.get("local_branch.patch_embed.weight")
    if lpe is not None:
        if lpe.ndim == 4:
            out_dim, in_ch, kH, kW = lpe.shape
        else:
            out_dim, flat = lpe.shape
            kH = kW = 16
            in_ch = flat // (kH * kW)
        cfg["local_embed_dim"] = int(out_dim)
        cfg["local_channels"] = int(in_ch)
        cfg["local_patch_size"] = int(kH)
        cfg["local_img_size"] = (512, 512)

    # Depths from block keys.
    g_depth = max(
        (int(k.split(".")[2]) for k in sd if k.startswith("global_branch.blocks.")),
        default=-1,
    ) + 1
    l_depth = max(
        (int(k.split(".")[2]) for k in sd if k.startswith("local_branch.blocks.")),
        default=-1,
    ) + 1
    if g_depth > 0:
        cfg["global_depth"] = g_depth
    if l_depth > 0:
        cfg["local_depth"] = l_depth

    # Sensible defaults.
    cfg.setdefault("global_num_heads", 12)
    cfg.setdefault("local_num_heads", 8)
    cfg.setdefault("cross_num_heads", 8)
    cfg.setdefault("cross_layers", 2)
    cfg.setdefault("decoder_depth", 2)
    cfg.setdefault("out_channels", 1)
    cfg.setdefault("mlp_ratio", 4.0)
    cfg.setdefault("drop_rate", 0.0)
    cfg.setdefault("drop_path", 0.0)
    cfg.setdefault("global_region_h", 7)
    cfg.setdefault("global_region_w", 7)

    return {"model": cfg, "_inferred": True}


def from_huggingface_hub(repo_id: str, cache_dir: Path | None = None,
                          token: str | None = None) -> Checkpoint:
    """Download a checkpoint from the HuggingFace Hub and return it.

    Requires the optional ``huggingface_hub`` package, which is installed
    by the base ``cranpm`` distribution. ``token`` falls back to
    :func:`cranpm.config.get_hf_token`.
    """
    from huggingface_hub import snapshot_download

    from cranpm.config import get_hf_token, get_paths

    cache_dir = cache_dir or get_paths().checkpoints_dir / repo_id.replace("/", "__")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = token or get_hf_token()
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        token=token,
        allow_patterns=["*.json", "*.safetensors", "*.ckpt", "README.md"],
    )
    return Checkpoint.from_path(Path(local_dir))
