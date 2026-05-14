import torch
import torch.nn as nn

from .global_branch import GlobalBranch
from .local_branch import LocalBranch
from .cross_attention import CrossAttentionBridge
from .prediction_head import CNNDecoder


class MultiScaleTopoFlow(nn.Module):
    """Multi-Scale TopoFlow for Europe PM2.5 super-resolution forecasting.

    Two-branch architecture:
        Global branch (0.25deg): ERA5 over full Europe → (B, N_g, D_g)
        Local branch (0.01deg):  GHAP 512x512 patch   → (B, N_l, D_l) + skip
        Cross-attention bridge:  local queries, global keys/values
        CNN decoder:             tokens + skip → pixel-level PM2.5

    Forward pass:
        1. Global branch encodes ERA5 + elevation + lead_time
        2. Local branch encodes GHAP patch + hires elevation → tokens + skip
        3. Cross-attention fuses global context into local tokens
        4. CNN decoder progressively upsamples (32x32→512x512) with skip
    """

    def __init__(
        self,
        # Global branch
        era5_channels: int = 30,
        global_img_size: tuple = (168, 280),
        global_patch_size: int = 8,
        global_embed_dim: int = 768,
        global_depth: int = 8,
        global_num_heads: int = 12,
        # Local branch
        local_channels: int = 2,
        local_img_size: tuple = (512, 512),
        local_patch_size: int = 16,
        local_embed_dim: int = 512,
        local_depth: int = 6,
        local_num_heads: int = 8,
        # Cross-attention
        cross_num_heads: int = 8,
        cross_layers: int = 2,
        # Prediction
        decoder_depth: int = 2,  # kept for config compat, not used by CNN decoder
        out_channels: int = 1,
        # Shared
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        drop_path: float = 0.1,
        # Regional wind scanning
        global_region_h: int = 7,
        global_region_w: int = 7,
    ):
        super().__init__()

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

        global_grid_h = global_img_size[0] // global_patch_size
        global_grid_w = global_img_size[1] // global_patch_size

        self.cross_attention = CrossAttentionBridge(
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
            skip_channels=local_embed_dim,  # Skip from local patch_embed
        )

    def forward(self, era5, elevation_coarse, ghap_patch, elevation_hires, lead_time,
                patch_center=None, wind_at_patch=None):
        """
        Args:
            era5:             (B, C, H_g, W_g) ERA5+CAMS fields (C=35)
            elevation_coarse: (B, H_g, W_g) coarse elevation for global TopoFlow
            ghap_patch:       (B, 4, 512, 512) [PM2.5, elevation, lat, lon]
            elevation_hires:  (B, 512, 512) hires elevation for local TopoFlow
            lead_time:        (B,) forecast horizon in days
            patch_center:     (B, 2) lat/lon of local patch center (optional)
            wind_at_patch:    (B, 2) u10/v10 wind at local patch (optional)

        Returns:
            pred: (B, 1, 512, 512) PM2.5 prediction at 0.01deg (normalized)
        """
        # 1. Global: ERA5 → global context features
        global_feats = self.global_branch(era5, elevation_coarse, lead_time)

        # 2. Local: GHAP patch → local features + skip for decoder
        local_feats, skip = self.local_branch(ghap_patch, elevation_hires)

        # 3. Wind-guided cross-attention: local queries attend to global context
        fused = self.cross_attention(
            local_feats, global_feats,
            patch_center=patch_center,
            wind_at_patch=wind_at_patch,
        )

        # 4. CNN decoder: progressive upsample with skip connection
        delta = self.prediction_head(fused, skip=skip)

        # 5. Delta prediction: decoder predicts CHANGE from day J,
        #    spatial structure comes free from GHAP input.
        #    ghap_patch[:, 0:1] is normalized PM2.5 for day J.
        ghap_today = ghap_patch[:, 0:1, :, :]  # (B, 1, 512, 512)
        pred = ghap_today + delta

        return pred

    def get_param_groups(self, lr_global=1e-4, lr_local=1e-4, lr_cross=1e-4, lr_head=1e-4):
        """Return parameter groups with per-component learning rates."""
        return [
            {"params": self.global_branch.parameters(), "lr": lr_global},
            {"params": self.local_branch.parameters(), "lr": lr_local},
            {"params": self.cross_attention.parameters(), "lr": lr_cross},
            {"params": self.prediction_head.parameters(), "lr": lr_head},
        ]
