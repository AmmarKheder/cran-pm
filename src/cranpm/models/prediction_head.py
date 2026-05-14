import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv3x3 → LeakyReLU → Conv3x3 + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class PixelShuffleUpBlock(nn.Module):
    """PixelShuffle upsample: Conv → PixelShuffle(2) → LeakyReLU → ResBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Conv produces 4x output channels, PixelShuffle rearranges to 2x spatial
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.resblock = ResBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)  # (B, out_ch, 2H, 2W)
        x = self.act(x)
        x = self.resblock(x)
        return x


class CNNDecoder(nn.Module):
    """PixelShuffle decoder: tokens (32x32) → pixel-level PM2.5 (512x512).

    Uses PixelShuffle (sub-pixel convolution) + ResBlocks for high-quality
    upsampling without checkerboard artifacts.

    Architecture (4 upsample stages, each x2):
        (B, D, 32, 32) → (B, 256, 64, 64) → (B, 128, 128, 128)
        → (B, 64, 256, 256) → (B, 32, 512, 512) → (B, 1, 512, 512)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        grid_h: int = 32,
        grid_w: int = 32,
        out_channels: int = 1,
        skip_channels: int = 0,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_channels = out_channels

        # If skip connection, fuse with tokens before decoding
        in_ch = embed_dim + skip_channels if skip_channels > 0 else embed_dim
        if in_ch != embed_dim:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.skip_fuse = nn.Identity()

        # Progressive PixelShuffle upsample: 32 → 64 → 128 → 256 → 512
        channels = [embed_dim, 256, 128, 64, 32]

        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(PixelShuffleUpBlock(channels[i], channels[i + 1]))

        # Final projection to output channels
        self.final = nn.Conv2d(channels[-1], out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Near-zero init for delta prediction:
        # Small random weights (not exact zero) so gradients can flow
        # through to upstream layers (global/local/cross-attention).
        # With zeros, ∂loss/∂features = W_final = 0 → upstream frozen.
        # With std=0.01, delta ≈ 0.06 normalized (≈ 1.1 µg/m³) at init,
        # RMSE increases by ~0.07 but all layers receive gradients from step 1.
        nn.init.normal_(self.final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.final.bias)

    def forward(self, x, skip=None):
        """
        Args:
            x: (B, N, D) fused local features, N = grid_h * grid_w
            skip: (B, D_skip, grid_h, grid_w) optional skip from local branch

        Returns:
            pred: (B, out_channels, H, W) where H = grid_h * 16 = 512
        """
        B, N, D = x.shape

        # Reshape tokens to spatial grid
        x = x.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)

        # Fuse skip connection if provided
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fuse(x)

        # Progressive PixelShuffle upsample: 32 → 64 → 128 → 256 → 512
        for stage in self.stages:
            x = stage(x)

        # Final 1x1 conv
        x = self.final(x)

        return x


# Keep old class name for backward compatibility in tests
PredictionHead = CNNDecoder
