import math

import numpy as np
import torch
import torch.nn.functional as F

from .scan_orders import wind_band_hilbert


class WindScanner:
    """Wind-direction-aware patch reordering with precomputed sector orders.

    DEPRECATED: Use RegionalWindScanner for physically correct local wind.
    This class averages wind over the entire domain — incorrect for large grids
    where wind direction varies spatially (e.g., Europe).
    """

    def __init__(self, grid_h: int, grid_w: int, num_sectors: int = 16, device="cpu"):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.num_sectors = num_sectors
        self.device = device

        self.sector_orders = {}
        self.sector_inverse = {}
        self._precompute_sector_orders()

    def _precompute_sector_orders(self):
        for sector in range(self.num_sectors):
            angle = 2 * math.pi * sector / self.num_sectors
            order = wind_band_hilbert(self.grid_h, self.grid_w, angle)
            order_t = torch.tensor(order, dtype=torch.long, device=self.device)
            self.sector_orders[sector] = order_t

            # Precompute inverse for O(1) unscrambling
            inv = torch.empty_like(order_t)
            inv[order_t] = torch.arange(len(order), dtype=torch.long, device=self.device)
            self.sector_inverse[sector] = inv

    def _get_wind_sector(self, u_wind, v_wind):
        """Compute dominant wind sector from u/v fields."""
        # Average over spatial dims to get mean wind direction
        u_mean = u_wind.mean(dim=(-2, -1))
        v_mean = v_wind.mean(dim=(-2, -1))

        angle = torch.atan2(v_mean, u_mean)
        angle = (angle + 2 * math.pi) % (2 * math.pi)
        sector = (angle / (2 * math.pi) * self.num_sectors).long() % self.num_sectors
        return sector

    def reorder(self, tokens, u_wind, v_wind):
        """Reorder tokens along wind direction.

        Args:
            tokens: (B, N, D) patch tokens
            u_wind: (B, H, W) u-component of wind
            v_wind: (B, H, W) v-component of wind

        Returns:
            reordered: (B, N, D) reordered tokens
            sectors: (B,) sector indices for inverse_reorder
        """
        B, N, D = tokens.shape
        sectors = self._get_wind_sector(u_wind, v_wind)

        reordered = torch.zeros_like(tokens)
        for b in range(B):
            s = sectors[b].item()
            order = self.sector_orders[s]
            reordered[b] = tokens[b, order]

        return reordered, sectors

    def inverse_reorder(self, tokens, sectors):
        """Undo wind-direction reordering.

        Args:
            tokens: (B, N, D) reordered tokens
            sectors: (B,) sector indices from reorder()

        Returns:
            original_order: (B, N, D) tokens in original spatial order
        """
        B, N, D = tokens.shape
        restored = torch.zeros_like(tokens)
        for b in range(B):
            s = sectors[b].item()
            inv = self.sector_inverse[s]
            restored[b] = tokens[b, inv]

        return restored

    def reorder_like(self, tensor, sectors):
        """Apply same reordering to another tensor (coords, elevation).

        Args:
            tensor: (B, N) or (B, N, D) — must have same N as tokens
            sectors: from reorder()
        """
        is_2d = tensor.dim() == 2
        if is_2d:
            tensor = tensor.unsqueeze(-1)
        B, N, D = tensor.shape
        out = torch.zeros_like(tensor)
        for b in range(B):
            s = sectors[b].item()
            order = self.sector_orders[s]
            out[b] = tensor[b, order]
        return out.squeeze(-1) if is_2d else out

    def to(self, device):
        self.device = device
        for sector in self.sector_orders:
            self.sector_orders[sector] = self.sector_orders[sector].to(device)
            self.sector_inverse[sector] = self.sector_inverse[sector].to(device)
        return self


class RegionalWindScanner:
    """Wind scanning with LOCAL wind direction per sub-region.

    Physically correct alternative to WindScanner: instead of averaging wind
    over the entire domain (physically wrong for Europe-scale), divides the
    patch grid into local regions and reorders tokens within each region
    using the LOCAL wind direction.

    Physical motivation:
        Wind is not uniform across Europe. On a given day:
        - Iberia: westerlies from the Atlantic
        - Po Valley: calm / trapped by Alps
        - Scandinavia: northerlies from Arctic
        - Eastern Europe: continental easterlies
        Averaging these directions destroys the local transport signal.

    For a 21x35 grid with 7x7 regions: 3x5 = 15 regions of 49 tokens each.
    Each region uses wind_band_hilbert to reorder its tokens along the
    local wind direction.

    Args:
        grid_h, grid_w: full patch grid size (e.g., 21, 35)
        region_h, region_w: region size in patches (e.g., 7, 7)
        num_sectors: angular discretization (default 16 = 22.5 deg bins)
        device: torch device
    """

    def __init__(self, grid_h, grid_w, region_h=7, region_w=7,
                 num_sectors=16, device="cpu", shuffle_mode="wind",
                 random_seed=0):
        """Args:
            shuffle_mode: one of
              - "wind"   (default): wind-direction-aligned ordering (CRAN-PM v3 default)
              - "random": fixed pseudo-random permutation per region (seeded);
                          ignores u/v_wind, returns the same permutation every call
              - "raster": identity (no reorder, tokens stay row-major).
            random_seed: only used when shuffle_mode='random'.
        """
        assert shuffle_mode in ("wind", "random", "raster"), \
            f"unknown shuffle_mode: {shuffle_mode}"
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.region_h = region_h
        self.region_w = region_w
        self.num_sectors = num_sectors
        self.num_patches = grid_h * grid_w
        self.device = device
        self.shuffle_mode = shuffle_mode
        self.random_seed = random_seed

        # Build region definitions
        self.regions = self._build_regions()
        self.n_regions = len(self.regions)

        # Precompute scan orders for all unique region sizes x sectors
        self._scan_cache = {}  # (rh, rw, sector) -> (order_tensor, inverse_tensor)
        self._precompute_all()

    def _build_regions(self):
        """Build list of regions. Last row/col absorbs remainder if grid
        doesn't divide evenly by region size."""
        regions = []

        # Row boundaries
        row_starts = list(range(0, self.grid_h, self.region_h))
        col_starts = list(range(0, self.grid_w, self.region_w))

        for i, r0 in enumerate(row_starts):
            # Last region row absorbs remainder
            r1 = row_starts[i + 1] if i + 1 < len(row_starts) else self.grid_h
            rh = r1 - r0

            for j, c0 in enumerate(col_starts):
                c1 = col_starts[j + 1] if j + 1 < len(col_starts) else self.grid_w
                rw = c1 - c0

                # Global indices of tokens in this region (raster scan)
                indices = []
                for ri in range(rh):
                    for ci in range(rw):
                        global_idx = (r0 + ri) * self.grid_w + (c0 + ci)
                        indices.append(global_idx)

                regions.append({
                    "indices": torch.tensor(indices, dtype=torch.long, device=self.device),
                    "rh": rh,
                    "rw": rw,
                    "r0": r0,
                    "c0": c0,
                })

        return regions

    def _precompute_all(self):
        """Precompute the ordering tensor for every (rh, rw, sector).

        ``shuffle_mode`` selects what the order means:
            wind   → wind_band_hilbert(rh, rw, sector_angle)
            random → a fixed pseudo-random permutation (sector ignored)
            raster → identity (sector ignored)

        We still write to every sector slot so the rest of the codepath
        (which indexes by sector) is mode-agnostic.
        """
        sizes = set((r["rh"], r["rw"]) for r in self.regions)
        rng = (np.random.default_rng(self.random_seed)
               if self.shuffle_mode == "random" else None)
        for rh, rw in sizes:
            n = rh * rw
            for s in range(self.num_sectors):
                if self.shuffle_mode == "wind":
                    angle = 2 * math.pi * s / self.num_sectors
                    order = wind_band_hilbert(rh, rw, angle)
                elif self.shuffle_mode == "random":
                    order = list(range(n))
                    rng.shuffle(order)
                else:  # raster
                    order = list(range(n))
                order_t = torch.tensor(order, dtype=torch.long, device=self.device)
                inv_t = torch.empty_like(order_t)
                inv_t[order_t] = torch.arange(len(order), dtype=torch.long, device=self.device)
                self._scan_cache[(rh, rw, s)] = (order_t, inv_t)

    def _get_regional_sectors(self, u_wind, v_wind):
        """Compute wind sector for each region from local u/v wind.

        Args:
            u_wind: (B, H, W) u-component at ERA5 pixel resolution
            v_wind: (B, H, W) v-component at ERA5 pixel resolution

        Returns:
            sectors: (B, n_regions) sector index per region
        """
        B, H, W = u_wind.shape
        ppH = H / self.grid_h  # pixels per patch row
        ppW = W / self.grid_w  # pixels per patch col

        sectors = torch.zeros(B, self.n_regions, dtype=torch.long, device=u_wind.device)

        for idx, reg in enumerate(self.regions):
            # Pixel extent of this region in the wind field
            pr0 = int(reg["r0"] * ppH)
            pr1 = min(int((reg["r0"] + reg["rh"]) * ppH), H)
            pc0 = int(reg["c0"] * ppW)
            pc1 = min(int((reg["c0"] + reg["rw"]) * ppW), W)

            u_local = u_wind[:, pr0:pr1, pc0:pc1].mean(dim=(-2, -1))
            v_local = v_wind[:, pr0:pr1, pc0:pc1].mean(dim=(-2, -1))

            angle = torch.atan2(v_local, u_local)
            angle = (angle + 2 * math.pi) % (2 * math.pi)
            sec = (angle / (2 * math.pi) * self.num_sectors).long() % self.num_sectors
            sectors[:, idx] = sec

        return sectors

    def reorder(self, tokens, u_wind, v_wind):
        """Reorder tokens within each region by local wind direction.

        Args:
            tokens: (B, N, D) patch tokens in spatial order
            u_wind: (B, H, W) u-component wind field
            v_wind: (B, H, W) v-component wind field

        Returns:
            reordered: (B, N, D) tokens reordered within each region
            sectors: (B, n_regions) sector per region (for inverse)
        """
        B, N, D = tokens.shape
        sectors = self._get_regional_sectors(u_wind, v_wind)

        reordered = tokens.clone()
        for b in range(B):
            for r_idx, reg in enumerate(self.regions):
                s = sectors[b, r_idx].item()
                gidx = reg["indices"]
                order, _ = self._scan_cache[(reg["rh"], reg["rw"], s)]
                reordered[b, gidx] = tokens[b, gidx[order]]

        return reordered, sectors

    def reorder_like(self, tensor, sectors):
        """Apply same regional reordering to another tensor (coords, elevation).

        Args:
            tensor: (B, N) or (B, N, D) — same N as tokens
            sectors: (B, n_regions) from reorder()

        Returns:
            reordered tensor with same shape
        """
        is_2d = tensor.dim() == 2
        if is_2d:
            tensor = tensor.unsqueeze(-1)

        B = tensor.shape[0]
        out = tensor.clone()
        for b in range(B):
            for r_idx, reg in enumerate(self.regions):
                s = sectors[b, r_idx].item()
                gidx = reg["indices"]
                order, _ = self._scan_cache[(reg["rh"], reg["rw"], s)]
                out[b, gidx] = tensor[b, gidx[order]]

        return out.squeeze(-1) if is_2d else out

    def inverse_reorder(self, tokens, sectors):
        """Undo regional wind reordering.

        Args:
            tokens: (B, N, D) reordered tokens
            sectors: (B, n_regions) from reorder()

        Returns:
            restored: (B, N, D) tokens in original spatial order
        """
        B, N, D = tokens.shape
        restored = tokens.clone()
        for b in range(B):
            for r_idx, reg in enumerate(self.regions):
                s = sectors[b, r_idx].item()
                gidx = reg["indices"]
                _, inv = self._scan_cache[(reg["rh"], reg["rw"], s)]
                restored[b, gidx] = tokens[b, gidx[inv]]

        return restored

    def to(self, device):
        """Move all tensors to device."""
        self.device = device
        for reg in self.regions:
            reg["indices"] = reg["indices"].to(device)
        new_cache = {}
        for key, (order, inv) in self._scan_cache.items():
            new_cache[key] = (order.to(device), inv.to(device))
        self._scan_cache = new_cache
        return self
