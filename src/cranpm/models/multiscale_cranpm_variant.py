"""
CRAN-PM variant model for ablation study.

Supports:
  - fusion_type ∈ {'fine_queries_coarse', 'no_cross_attn', 'concat_self_attn',
                    'feature_add', 'film', 'coarse_queries_fine', 'bidirectional'}
  - fine_only=True → only local branch, no global branch (Fine-only ViT baseline)
  - no_wind=True → disable wind reordering (zero ERA5 wind channels) and cross-attn wind bias
  - no_elevation=True → zero elevation inputs
  - no_temporal=True → zero previous-day channels

All variants share the same forward() signature as MultiScaleTopoFlow.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_branch import GlobalBranch
from .local_branch import LocalBranch
from .cross_attention_variants import FlexibleFusionBridge
from .prediction_head import CNNDecoder


class MultiScaleCRANPMVariant(nn.Module):
    """Ablation variant of CRAN-PM.

    Args:
        fusion_type: which cross-scale fusion strategy to use
        fine_only:   if True, skip global branch entirely (Fine-only ViT)
        no_wind:     if True, zero wind in ERA5 before global branch
        no_elevation: if True, zero elevation inputs
        no_temporal:  if True, zero previous-day channels in ERA5 + local input
    """

    def __init__(
        self,
        fusion_type: str = 'fine_queries_coarse',
        fine_only: bool = False,
        no_wind: bool = False,
        no_elevation: bool = False,
        no_temporal: bool = False,
        # Architecture (same as CRAN-PM full)
        era5_channels: int = 70,
        global_img_size: tuple = (168, 280),
        global_patch_size: int = 8,
        global_embed_dim: int = 768,
        global_depth: int = 8,
        global_num_heads: int = 12,
        local_channels: int = 5,
        local_img_size: tuple = (512, 512),
        local_patch_size: int = 16,
        local_embed_dim: int = 512,
        local_depth: int = 6,
        local_num_heads: int = 8,
        cross_num_heads: int = 8,
        cross_layers: int = 2,
        decoder_depth: int = 2,
        out_channels: int = 1,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
        global_region_h: int = 7,
        global_region_w: int = 7,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.fine_only = fine_only
        self.no_wind = no_wind
        self.no_elevation = no_elevation
        self.no_temporal = no_temporal

        # Always build local branch
        self.local_branch = LocalBranch(
            in_channels=local_channels,
            img_size=local_img_size,
            patch_size=local_patch_size,
            embed_dim=local_embed_dim,
            depth=local_depth,
            num_heads=local_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=drop_path,
        )

        # Global branch (built always; optionally disabled at forward time)
        self.global_branch = GlobalBranch(
            in_channels=era5_channels,
            img_size=global_img_size,
            patch_size=global_patch_size,
            embed_dim=global_embed_dim,
            depth=global_depth,
            num_heads=global_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path=drop_path,
            region_h=global_region_h,
            region_w=global_region_w,
        )

        global_grid_h = global_img_size[0] // global_patch_size
        global_grid_w = global_img_size[1] // global_patch_size

        # Flexible fusion bridge
        self.cross_attention = FlexibleFusionBridge(
            fusion_type=fusion_type,
            local_dim=local_embed_dim,
            global_dim=global_embed_dim,
            num_heads=cross_num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=drop_rate,
            drop_path=drop_path,
            num_layers=cross_layers,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
        )

        local_grid_h = local_img_size[0] // local_patch_size
        local_grid_w = local_img_size[1] // local_patch_size

        self.prediction_head = CNNDecoder(
            embed_dim=local_embed_dim,
            grid_h=local_grid_h,
            grid_w=local_grid_w,
            out_channels=out_channels,
            skip_channels=local_embed_dim,
        )

        self.era5_channels = era5_channels

    def forward(self, era5, elevation_coarse, ghap_patch, elevation_hires, lead_time,
                patch_center=None, wind_at_patch=None):
        B = era5.shape[0]
        device = era5.device

        # ── Ablation: zero temporal context (prev-day channels) ───────────────
        if self.no_temporal:
            # ERA5: channels [35:70] = ERA5_prev + CAMS_prev
            era5 = era5.clone()
            era5[:, 35:] = 0.0
            # Local: channel 4 = ghap_t-1
            ghap_patch = ghap_patch.clone()
            ghap_patch[:, 4:] = 0.0

        # ── Ablation: zero wind channels in ERA5 ──────────────────────────────
        if self.no_wind:
            era5 = era5.clone() if not self.no_temporal else era5
            era5[:, 0] = 0.0   # u10
            era5[:, 1] = 0.0   # v10
            wind_at_patch = None  # disable cross-attn wind bias too

        # ── Ablation: zero elevation inputs ───────────────────────────────────
        if self.no_elevation:
            elevation_coarse = torch.zeros_like(elevation_coarse)
            elevation_hires  = torch.zeros_like(elevation_hires)
            # Also zero elevation channel in local input (channel 1)
            ghap_patch = ghap_patch.clone()
            ghap_patch[:, 1] = 0.0

        # 1. Local branch always runs
        local_feats, skip = self.local_branch(ghap_patch, elevation_hires)

        # 2. Global branch (skip if fine_only)
        if self.fine_only or self.fusion_type == 'no_cross_attn':
            # For fine_only: use zeros as global context (model never sees global)
            # For no_cross_attn: global still computed but fusion is skipped
            if self.fine_only:
                global_feats = torch.zeros(B, 1, self.global_branch.embed_dim, device=device)
            else:
                global_feats = self.global_branch(era5, elevation_coarse, lead_time)
        else:
            global_feats = self.global_branch(era5, elevation_coarse, lead_time)

        # 3. Fusion
        if self.fine_only:
            fused = local_feats  # skip fusion entirely
        else:
            fused = self.cross_attention(
                local_feats, global_feats,
                patch_center=patch_center,
                wind_at_patch=wind_at_patch,
            )

        # 4. Decoder + delta
        delta = self.prediction_head(fused, skip=skip)
        ghap_today = ghap_patch[:, 0:1, :, :]
        pred = ghap_today + delta
        return pred

    def get_param_groups(self, lr_global=1e-4, lr_local=1e-4, lr_cross=1e-4, lr_head=1e-4):
        return [
            {"params": self.global_branch.parameters(), "lr": lr_global},
            {"params": self.local_branch.parameters(), "lr": lr_local},
            {"params": self.cross_attention.parameters(), "lr": lr_cross},
            {"params": self.prediction_head.parameters(), "lr": lr_head},
        ]
