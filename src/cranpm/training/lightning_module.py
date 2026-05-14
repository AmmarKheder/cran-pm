import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..models.multiscale_topoflow import MultiScaleTopoFlow
from .loss import MultiScaleLoss
from .visualize import log_prediction_maps


class MultiScaleTopoFlowLightning(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        mc = config["model"]
        self.model = MultiScaleTopoFlow(
            era5_channels=mc.get("era5_channels", 30),
            global_img_size=tuple(mc.get("global_img_size", [168, 280])),
            global_patch_size=mc.get("global_patch_size", 8),
            global_embed_dim=mc.get("global_embed_dim", 768),
            global_depth=mc.get("global_depth", 8),
            global_num_heads=mc.get("global_num_heads", 12),
            local_channels=mc.get("local_channels", 2),
            local_img_size=tuple(mc.get("local_img_size", [512, 512])),
            local_patch_size=mc.get("local_patch_size", 16),
            local_embed_dim=mc.get("local_embed_dim", 512),
            local_depth=mc.get("local_depth", 6),
            local_num_heads=mc.get("local_num_heads", 8),
            cross_num_heads=mc.get("cross_num_heads", 8),
            cross_layers=mc.get("cross_layers", 2),
            decoder_depth=mc.get("decoder_depth", 2),
            out_channels=mc.get("out_channels", 1),
            mlp_ratio=mc.get("mlp_ratio", 4.0),
            drop_rate=mc.get("drop_rate", 0.1),
            drop_path=mc.get("drop_path", 0.1),
            global_region_h=mc.get("global_region_h", 7),
            global_region_w=mc.get("global_region_w", 7),
        )

        # Zero-init patch embedding weights for NEW t-1 channels.
        # Without this, random projections of ERA5(t-1) and GHAP(t-1)
        # inject noise with magnitude ≈ signal, destroying the SNR
        # and preventing the model from learning useful deltas.
        # With zero-init, the model starts as v7 (t-1 channels contribute
        # nothing) and gradually learns to incorporate temporal context.
        n_era5_base = 35   # 30 ERA5(t) + 5 CAMS(t)
        n_local_base = 4   # ghap_t, elev, lat, lon
        era5_ch = mc.get("era5_channels", 30)
        local_ch = mc.get("local_channels", 2)
        global_ps = mc.get("global_patch_size", 8)
        local_ps = mc.get("local_patch_size", 16)
        with torch.no_grad():
            # patch_embed is nn.Linear(C*K*K, D) — zero columns for t-1 channels
            # F.unfold arranges features as [ch0_px0..ch0_pxK², ch1_px0..., ...]
            # so channel c occupies columns [c*K², (c+1)*K²)
            if era5_ch > n_era5_base:
                k_sq = global_ps * global_ps
                self.model.global_branch.patch_embed.weight[:, n_era5_base * k_sq:] = 0.0
            if local_ch > n_local_base:
                k_sq = local_ps * local_ps
                self.model.local_branch.patch_embed.weight[:, n_local_base * k_sq:] = 0.0

        self.criterion = MultiScaleLoss(
            alpha_mse=config["train"].get("alpha_mse", 1.0),
            alpha_ssim=config["train"].get("alpha_ssim", 0.0),
            alpha_grad=config["train"].get("alpha_grad", 0.1),
            alpha_spectral=config["train"].get("alpha_spectral", 0.0),
            alpha_station=config["train"].get("alpha_station", 0.0),
            ghap_mean=config["data"].get("ghap_mean", 15.0),
            ghap_std=config["data"].get("ghap_std", 20.0),
            underestimate_penalty=config["train"].get("underestimate_penalty", 2.0),
            ffl_alpha=config["train"].get("ffl_alpha", 1.0),
        )

        # GHAP stats for denormalization in logging
        self.ghap_mean = config["data"].get("ghap_mean", 15.0)
        self.ghap_std = config["data"].get("ghap_std", 20.0)

    def forward(self, batch):
        return self.model(
            era5=batch["era5"],
            elevation_coarse=batch["elevation_coarse"],
            ghap_patch=batch["local_input"],
            elevation_hires=batch["elevation_hires"],
            lead_time=batch["lead_time"],
            patch_center=batch.get("patch_center"),
            wind_at_patch=batch.get("wind_at_patch"),
        )

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch["target"]

        # NaN detection: catch overflow immediately
        if torch.isnan(pred).any():
            nan_pct = torch.isnan(pred).float().mean().item() * 100
            era5 = batch["era5"]
            local_in = batch["local_input"]
            print(f"[RANK {self.global_rank}] NaN in pred at step {self.global_step}: "
                  f"{nan_pct:.1f}% NaN | era5_nan={torch.isnan(era5).any().item()} "
                  f"era5_max={era5.abs().max().item():.1f} "
                  f"local_nan={torch.isnan(local_in).any().item()} "
                  f"local_max={local_in.abs().max().item():.1f}", flush=True)
            # Check which params have NaN
            for name, p in self.model.named_parameters():
                if torch.isnan(p).any():
                    print(f"  NaN param: {name}", flush=True)
            # Zero loss connected to params (NOT pred*0 — NaN*0=NaN in IEEE 754)
            return sum(p.sum() * 0.0 for p in self.model.parameters())

        loss, metrics = self.criterion(
            pred, target,
            station_pixels=batch.get("station_pixels"),
            station_values=batch.get("station_values"),
            station_count=batch.get("station_count"),
        )

        # NaN loss detection
        if torch.isnan(loss):
            print(f"[RANK {self.global_rank}] NaN loss at step {self.global_step}", flush=True)
            return sum(p.sum() * 0.0 for p in self.model.parameters())

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/mse", metrics["mse"], sync_dist=True)
        if "grad_loss" in metrics:
            self.log("train/grad_loss", metrics["grad_loss"], sync_dist=True)
        if "spectral_loss" in metrics:
            self.log("train/spectral_loss", metrics["spectral_loss"], sync_dist=True)
        if "station_loss" in metrics:
            self.log("train/station_loss", metrics["station_loss"], sync_dist=True)
        if "land_pct" in metrics:
            self.log("train/land_pct", metrics["land_pct"], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch["target"]

        loss, metrics = self.criterion(
            pred, target,
            station_pixels=batch.get("station_pixels"),
            station_values=batch.get("station_values"),
            station_count=batch.get("station_count"),
        )

        # Compute RMSE in original units
        rmse = self._compute_rmse(pred, target)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mse", metrics["mse"], sync_dist=True)
        self.log("val/rmse", rmse, prog_bar=True, sync_dist=True)

        # Per-horizon RMSE
        horizons = batch["lead_time"]
        for h_val in horizons.unique():
            mask = horizons == h_val
            if mask.any():
                h_rmse = self._compute_rmse(pred[mask], target[mask])
                self.log(f"val/rmse_T{int(h_val.item())}", h_rmse, sync_dist=True)

        # Log visualization on first batch, rank 0
        if batch_idx == 0 and self.global_rank == 0 and self.logger is not None:
            log_prediction_maps(
                self.logger, pred, target,
                self.ghap_mean, self.ghap_std,
                step=self.global_step,
            )

        return loss

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        target = batch["target"]

        loss, metrics = self.criterion(pred, target)
        rmse = self._compute_rmse(pred, target)

        self.log("test/loss", loss, sync_dist=True)
        self.log("test/rmse", rmse, sync_dist=True)

        return {"loss": loss, "rmse": rmse}

    def _compute_rmse(self, pred, target):
        """RMSE in original ug/m3 units."""
        # Cast to float32 for numerical stability (bf16-safe)
        pred_f = torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target_f = target.float()

        # Denormalize
        pred_orig = pred_f * self.ghap_std + self.ghap_mean
        target_orig = target_f * self.ghap_std + self.ghap_mean

        mask = (target_orig > 0) & torch.isfinite(target_orig) & torch.isfinite(pred_orig)
        mask = mask.float()
        denom = mask.sum()
        if denom < 1:
            return torch.tensor(0.0, device=pred.device)

        mse = ((pred_orig - target_orig) ** 2 * mask).sum() / denom
        return torch.sqrt(mse)

    def configure_optimizers(self):
        tc = self.config["train"]
        lr = tc.get("learning_rate", 1e-4)
        wd = tc.get("weight_decay", 0.05)
        warmup_epochs = tc.get("warmup_epochs", 0)
        epochs = tc.get("epochs", 200)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=tc.get("min_lr", 1e-6),
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=tc.get("min_lr", 1e-6),
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
