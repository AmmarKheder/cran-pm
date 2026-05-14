import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    """Land-masked loss with spatial gradient penalty for PM2.5 forecasting.

    Components:
        - Pixel MSE only on land pixels (target > 0 and finite)
        - Spatial gradient penalty: ||grad(pred) - grad(target)||^2 on land
          Forces the model to reproduce spatial detail / texture, not just
          minimize global MSE (which favors smooth/flat predictions).
        - Optional structural similarity (SSIM) component

    The gradient penalty is key for the CNN decoder: without it the decoder
    can satisfy MSE with smooth interpolation. The gradient penalty forces
    it to reconstruct edges and fine-grained spatial variation.
    """

    def __init__(
        self,
        alpha_mse: float = 1.0,
        alpha_ssim: float = 0.0,
        alpha_grad: float = 0.1,
        alpha_spectral: float = 0.0,
        alpha_station: float = 0.0,
        ghap_mean: float = 15.0,
        ghap_std: float = 20.0,
        hotspot_alpha: float = 2.0,
        hotspot_threshold: float = 0.25,   # normalized: (20-15)/20 = 0.25 → raw 20 µg/m³
        hotspot_scale: float = 0.5,
        hotspot_max_weight: float = 5.0,
        underestimate_penalty: float = 2.0,  # penalize pred < target this much more
        ffl_alpha: float = 1.0,  # Focal Frequency Loss exponent (ICCV 2021)
    ):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_ssim = alpha_ssim
        self.alpha_grad = alpha_grad
        self.alpha_spectral = alpha_spectral
        self.alpha_station = alpha_station
        self.ffl_alpha = ffl_alpha
        # Threshold for land mask in normalized space:
        # raw > 0 → (raw - mean) / std > -mean/std
        self.land_threshold = -ghap_mean / ghap_std  # -0.75

        # Hotspot weighting: upweight polluted pixels
        self.hotspot_alpha = hotspot_alpha
        self.hotspot_threshold = hotspot_threshold
        self.hotspot_scale = hotspot_scale
        self.hotspot_max_weight = hotspot_max_weight
        self.underestimate_penalty = underestimate_penalty

        # Sobel kernels for spatial gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _land_mask(self, target):
        """Create land mask: valid land pixels where raw PM2.5 > 0.

        Target is normalized: (raw - mean) / std.
        Ocean = NaN (caught by isfinite).
        Land with raw > 0 means normalized > -mean/std = -0.75.
        """
        return (target > self.land_threshold) & torch.isfinite(target)

    def _pixel_weights(self, target):
        """Hotspot weighting: upweight polluted pixels (normalized space).

        weight = 1.0 + alpha * clamp((target - threshold) / scale, 0, max_weight)
          pixel at  5 µg/m³ (norm=-0.50): weight ≈ 1.0
          pixel at 20 µg/m³ (norm= 0.25): weight = 1.0  (threshold)
          pixel at 30 µg/m³ (norm= 0.75): weight = 3.0
          pixel at 40 µg/m³ (norm= 1.25): weight = 5.0
        """
        excess = (target - self.hotspot_threshold) / self.hotspot_scale
        excess = excess.clamp(min=0.0, max=self.hotspot_max_weight)
        return 1.0 + self.hotspot_alpha * excess

    def _masked_mse(self, pred, target, mask):
        """Asymmetric weighted MSE on masked (land) pixels.

        Under-estimation (pred < target) is penalized more than over-estimation.
        This prevents the model from compressing predictions toward the mean
        and forces it to preserve peak intensities at hotspots.

        Combined with hotspot weighting:
          - pixel at 30 µg/m³ under-estimated by 10 → loss × hotspot_weight × underestimate_penalty
          - pixel at 5 µg/m³ over-estimated by 2  → loss × 1.0 × 1.0
        """
        mask_f = mask.float()
        weights = self._pixel_weights(target) * mask_f

        # Asymmetric penalty: under-estimation gets extra weight
        under = (pred < target).float()  # 1 where pred < target
        asym = 1.0 + (self.underestimate_penalty - 1.0) * under  # 1.0 or underestimate_penalty
        weights = weights * asym

        denom = weights.sum().clamp(min=1.0)
        diff = (pred - target) ** 2
        return (diff * weights).sum() / denom

    def _spatial_gradient_loss(self, pred, target, mask):
        """Penalize difference in spatial gradients (Sobel) on land pixels.

        Forces the CNN decoder to reproduce edges and texture,
        not just smooth interpolation.
        """
        # Compute gradients with Sobel filter
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        tgt_gx = F.conv2d(target, self.sobel_x, padding=1)
        tgt_gy = F.conv2d(target, self.sobel_y, padding=1)

        # L2 difference of gradients, masked to land
        mask_f = mask.float()
        # Erode mask by 1 pixel (Sobel touches neighbors)
        mask_eroded = F.avg_pool2d(mask_f, 3, stride=1, padding=1)
        mask_eroded = (mask_eroded > 0.99).float()

        denom = mask_eroded.sum().clamp(min=1.0)

        grad_diff = (pred_gx - tgt_gx) ** 2 + (pred_gy - tgt_gy) ** 2
        return (grad_diff * mask_eroded).sum() / denom

    def _spectral_loss(self, pred, target, mask):
        """Focal Frequency Loss (FFL) — Lin et al., ICCV 2021.

        Adaptively focuses on hard-to-synthesize frequency components.
        Weight w(u,v) = |FFT(pred) - FFT(target)|^alpha makes the loss
        focus more on frequencies where the model has larger errors
        (typically high frequencies = edges, hotspots).

        FFL = mean( w * |FFT(pred) - FFT(target)|^2 )
        """
        # Zero out ocean/invalid pixels for clean FFT
        pred_clean = torch.where(mask, pred, torch.zeros_like(pred))
        target_clean = torch.where(mask, target, torch.zeros_like(target))

        # Replace any remaining NaN/Inf (safety for bf16)
        pred_clean = torch.nan_to_num(pred_clean, nan=0.0, posinf=0.0, neginf=0.0)
        target_clean = torch.nan_to_num(target_clean, nan=0.0, posinf=0.0, neginf=0.0)

        # 2D real FFT with ortho normalization (bf16 → float32)
        pred_fft = torch.fft.rfft2(pred_clean.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target_clean.float(), norm="ortho")

        # Complex difference → amplitude
        diff = pred_fft - target_fft
        diff_amp = torch.abs(diff)

        # Focal weight: w = |diff|^alpha (detach to avoid double gradient)
        # Clamp to prevent overflow when cubed (diff_amp^3 with alpha=1)
        weight = diff_amp.detach().clamp(max=10.0) ** self.ffl_alpha

        # Weighted MSE in frequency domain
        return (weight * diff_amp ** 2).mean()

    def _station_loss(self, pred, station_pixels, station_values, station_count):
        """Station loss: MSE at EEA ground station locations.

        Uses bilinear interpolation (grid_sample) to extract predicted values
        at sub-pixel station coordinates, then compares with actual measurements.

        Args:
            pred: (B, 1, H, W) model prediction (normalized)
            station_pixels: (B, MAX, 2) local pixel coords [row, col] in patch
            station_values: (B, MAX) normalized PM2.5 from EEA stations
            station_count: (B,) number of valid stations per sample
        """
        B, _, H, W = pred.shape
        total_loss = torch.tensor(0.0, device=pred.device)
        total_count = 0

        for b in range(B):
            n = station_count[b].item()
            if n == 0:
                continue

            # Get valid station data for this sample
            px = station_pixels[b, :n]   # (n, 2) = [row, col]
            sv = station_values[b, :n]   # (n,) normalized PM2.5

            # Convert pixel coords to grid_sample format: [-1, 1]
            # grid_sample expects (x=col, y=row) in [-1, 1]
            grid_x = (px[:, 1] / (W - 1)) * 2 - 1  # col → x
            grid_y = (px[:, 0] / (H - 1)) * 2 - 1  # row → y
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (n, 2)
            grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n, 2)

            # Bilinear interpolation at station locations
            pred_at_stations = F.grid_sample(
                pred[b:b+1],  # (1, 1, H, W)
                grid.to(pred.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, 1, 1, n)
            pred_vals = pred_at_stations.squeeze()  # (n,)

            total_loss = total_loss + ((pred_vals - sv.to(pred.dtype)) ** 2).sum()
            total_count += n

        if total_count == 0:
            return total_loss  # 0.0, but stays connected to graph via pred
        return total_loss / total_count

    def _ssim_loss(self, pred, target, window_size=11):
        """Simple SSIM-based loss (1 - SSIM)."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)

        channels = pred.shape[1]
        window = window.expand(channels, -1, -1, -1)
        pad = window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=channels) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

        ssim = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1 - ssim.mean()

    def forward(self, pred, target, station_pixels=None, station_values=None,
                station_count=None):
        """
        Args:
            pred:   (B, 1, H, W) predicted PM2.5
            target: (B, 1, H, W) ground truth PM2.5
            station_pixels: (B, MAX, 2) optional — station pixel coords in patch
            station_values: (B, MAX) optional — station PM2.5 (normalized)
            station_count:  (B,) optional — valid station count per sample

        Returns:
            loss: scalar
            metrics: dict with component losses
        """
        # Land mask
        mask = self._land_mask(target)

        # MSE on land pixels only
        mse = self._masked_mse(pred, target, mask)
        loss = self.alpha_mse * mse
        metrics = {"mse": mse.detach()}

        # Spatial gradient penalty on land
        if self.alpha_grad > 0:
            grad_loss = self._spatial_gradient_loss(pred, target, mask)
            loss = loss + self.alpha_grad * grad_loss
            metrics["grad_loss"] = grad_loss.detach()

        # Spectral loss (FFT amplitude)
        if self.alpha_spectral > 0:
            spectral_loss = self._spectral_loss(pred, target, mask)
            loss = loss + self.alpha_spectral * spectral_loss
            metrics["spectral_loss"] = spectral_loss.detach()

        # Station loss (EEA ground measurements)
        if self.alpha_station > 0 and station_pixels is not None:
            stn_loss = self._station_loss(pred, station_pixels, station_values,
                                          station_count)
            loss = loss + self.alpha_station * stn_loss
            metrics["station_loss"] = stn_loss.detach()

        # Optional SSIM
        if self.alpha_ssim > 0:
            ssim_loss = self._ssim_loss(pred, target)
            loss = loss + self.alpha_ssim * ssim_loss
            metrics["ssim_loss"] = ssim_loss.detach()

        metrics["loss"] = loss.detach()
        metrics["land_pct"] = mask.float().mean().detach()
        return loss, metrics
