"""
Cross-attention fusion variants for CRAN-PM ablation study.

Supported fusion_type values:
    'fine_queries_coarse'  : (ours)  local Q, global K/V  — our CRAN-PM method
    'no_cross_attn'        : (i)     no fusion at all
    'concat_self_attn'     : (ii)    concatenate local+global tokens, apply self-attn
    'feature_add'          : (iii)   add mean-pooled global to each local token
    'film'                 : (iv)    FiLM: scale+shift local by linear(mean_global)
    'coarse_queries_fine'  : (v)     global Q, local K/V — reverse direction
    'bidirectional'        : (vi)    both directions, outputs summed
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, Mlp
from timm.layers import DropPath

from .cross_attention import WindGuidedCrossAttentionLayer


# ── Shared helper ──────────────────────────────────────────────────────────────

def _make_cross_attn_layers(num_layers, dim, num_heads, mlp_ratio, drop, attn_drop, drop_path):
    return nn.ModuleList([
        WindGuidedCrossAttentionLayer(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
        )
        for _ in range(num_layers)
    ])


# ── Variant bridge ─────────────────────────────────────────────────────────────

class FlexibleFusionBridge(nn.Module):
    """Flexible cross-scale fusion that supports multiple strategies.

    All variants share the same interface as CrossAttentionBridge:
        forward(local_tokens, global_tokens, patch_center=None, wind_at_patch=None)
        → fused  (B, N_local, local_dim)
    """

    def __init__(
        self,
        fusion_type: str = 'fine_queries_coarse',
        local_dim: int = 512,
        global_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        num_layers: int = 2,
        global_grid_h: int = 21,
        global_grid_w: int = 35,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_layers = num_layers

        # Project global to local dim (used by most variants)
        self.global_proj = (
            nn.Linear(global_dim, local_dim)
            if global_dim != local_dim else nn.Identity()
        )

        # ── Strategy-specific layers ─────────────────────────────────────────

        if fusion_type in ('fine_queries_coarse', 'bidirectional'):
            # Standard cross-attn: local Q, global K/V
            self.layers = _make_cross_attn_layers(
                num_layers, local_dim, num_heads, mlp_ratio, drop, attn_drop, drop_path)
            self._register_global_coords(global_grid_h, global_grid_w)

        if fusion_type in ('coarse_queries_fine', 'bidirectional'):
            # Reverse cross-attn: global Q, local K/V
            # Note: output is in global (projected-to-local) dim, then mean-pooled back
            self.rev_layers = _make_cross_attn_layers(
                num_layers, local_dim, num_heads, mlp_ratio, drop, attn_drop, drop_path)

        if fusion_type == 'concat_self_attn':
            # Self-attn blocks over concatenated token sequence
            # After concat, only local-part is used as output
            self.self_attn_blocks = nn.ModuleList([
                Block(local_dim, num_heads, mlp_ratio, qkv_bias=True,
                      drop_path=drop_path, norm_layer=nn.LayerNorm)
                for _ in range(num_layers)
            ])

        if fusion_type == 'film':
            # FiLM conditioning: global mean → γ, β to scale+shift local norms
            self.film_norm = nn.LayerNorm(local_dim)
            self.film_scale = nn.Linear(local_dim, local_dim)  # γ
            self.film_shift = nn.Linear(local_dim, local_dim)  # β
            # init: scale → identity (weight=0, bias=1), shift → zero
            self.film_scale.weight.data.fill_(0)
            self.film_scale.bias.data.fill_(1)
            self.film_shift.weight.data.fill_(0)
            self.film_shift.bias.data.fill_(0)
            # Residual FFN after FiLM
            self.film_mlp = nn.ModuleList([
                Mlp(in_features=local_dim,
                    hidden_features=int(local_dim * mlp_ratio),
                    act_layer=nn.GELU, drop=drop)
                for _ in range(num_layers)
            ])
            self.film_mlp_norms = nn.ModuleList([
                nn.LayerNorm(local_dim) for _ in range(num_layers)
            ])

    def _register_global_coords(self, gh, gw):
        patch_deg = 8 * 0.25  # 2.0 degrees per patch
        lats = 72.0 - (torch.arange(gh).float() + 0.5) * patch_deg
        lons = -25.0 + (torch.arange(gw).float() + 0.5) * patch_deg
        lat_grid = lats.unsqueeze(1).expand(gh, gw).reshape(-1)
        lon_grid = lons.unsqueeze(0).expand(gh, gw).reshape(-1)
        global_coords = torch.stack([lat_grid, lon_grid], dim=-1)
        self.register_buffer("global_coords", global_coords, persistent=False)

    def _compute_wind_bias(self, patch_center, wind_at_patch):
        if patch_center is None or wind_at_patch is None:
            return None
        B = patch_center.shape[0]
        gc = self.global_coords.unsqueeze(0).expand(B, -1, -1)
        pc = patch_center.unsqueeze(1)
        direction = pc - gc
        dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = direction / dir_norm
        wind_dir = torch.stack([wind_at_patch[:, 1], wind_at_patch[:, 0]], dim=-1)
        wind_speed = wind_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        wind_dir = wind_dir / wind_speed
        alignment = (direction * wind_dir.unsqueeze(1)).sum(dim=-1)
        return alignment.unsqueeze(1).unsqueeze(1)

    def forward(self, local_tokens, global_tokens, patch_center=None, wind_at_patch=None):
        gp = self.global_proj(global_tokens)   # (B, N_g, D_l)

        ft = self.fusion_type

        # ── (i) No fusion ─────────────────────────────────────────────────────
        if ft == 'no_cross_attn':
            return local_tokens

        # ── (iii) Feature addition ────────────────────────────────────────────
        elif ft == 'feature_add':
            # Broadcast mean of projected global tokens to all local tokens
            global_mean = gp.mean(dim=1, keepdim=True)   # (B, 1, D_l)
            return local_tokens + global_mean

        # ── (iv) FiLM conditioning ────────────────────────────────────────────
        elif ft == 'film':
            global_mean = gp.mean(dim=1)                 # (B, D_l)
            x = local_tokens
            for mlp, norm in zip(self.film_mlp, self.film_mlp_norms):
                gamma = self.film_scale(global_mean).unsqueeze(1)   # (B, 1, D_l)
                beta  = self.film_shift(global_mean).unsqueeze(1)   # (B, 1, D_l)
                x = gamma * self.film_norm(x) + beta
                x = x + mlp(norm(x))
            return x

        # ── (ii) Concat + Self-Attn ───────────────────────────────────────────
        elif ft == 'concat_self_attn':
            N_l = local_tokens.shape[1]
            combined = torch.cat([local_tokens, gp], dim=1)  # (B, N_l+N_g, D)
            for blk in self.self_attn_blocks:
                combined = blk(combined)
            return combined[:, :N_l, :]   # return local part

        # ── (v) Coarse queries Fine ───────────────────────────────────────────
        elif ft == 'coarse_queries_fine':
            # Global tokens as Q, local tokens as K/V
            # Output: updated global (B, N_g, D_l), then mean-pool back to local
            x = gp
            for layer in self.rev_layers:
                x = layer(x, local_tokens, wind_bias=None)
            # Mean-pool enriched global and add as residual to local
            g_context = x.mean(dim=1, keepdim=True)    # (B, 1, D_l)
            return local_tokens + g_context

        # ── (vii) Fine queries Coarse (our method) ────────────────────────────
        elif ft == 'fine_queries_coarse':
            wind_bias = self._compute_wind_bias(patch_center, wind_at_patch)
            x = local_tokens
            for layer in self.layers:
                x = layer(x, gp, wind_bias=wind_bias)
            return x

        # ── (vi) Bidirectional ────────────────────────────────────────────────
        elif ft == 'bidirectional':
            # Forward: local Q, global K/V (our method)
            wind_bias = self._compute_wind_bias(patch_center, wind_at_patch)
            x_fwd = local_tokens
            for layer in self.layers:
                x_fwd = layer(x_fwd, gp, wind_bias=wind_bias)
            # Reverse: global Q, local K/V → mean-pool back
            x_rev = gp
            for layer in self.rev_layers:
                x_rev = layer(x_rev, local_tokens, wind_bias=None)
            g_context = x_rev.mean(dim=1, keepdim=True)
            return x_fwd + g_context

        else:
            raise ValueError(f"Unknown fusion_type: {ft!r}")
