"""
Lightning module for CRAN-PM ablation variant training.
Uses MultiScaleCRANPMVariant instead of MultiScaleTopoFlow.
"""
import torch
import pytorch_lightning as pl

from ..models.multiscale_cranpm_variant import MultiScaleCRANPMVariant
from .loss import MultiScaleLoss
from .visualize import log_prediction_maps


class CRANPMVariantLightning(pl.LightningModule):

    def __init__(self, config: dict, fusion_type: str = 'fine_queries_coarse',
                 fine_only: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.fusion_type = fusion_type
        self.fine_only = fine_only

        mc = config["model"]
        self.model = MultiScaleCRANPMVariant(
            fusion_type=fusion_type,
            fine_only=fine_only,
            era5_channels=mc.get("era5_channels", 70),
            global_img_size=tuple(mc.get("global_img_size", [168, 280])),
            global_patch_size=mc.get("global_patch_size", 8),
            global_embed_dim=mc.get("global_embed_dim", 768),
            global_depth=mc.get("global_depth", 8),
            global_num_heads=mc.get("global_num_heads", 12),
            local_channels=mc.get("local_channels", 5),
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

        self.criterion = MultiScaleLoss(
            alpha_mse=config["train"].get("alpha_mse", 1.0),
            alpha_ssim=config["train"].get("alpha_ssim", 0.0),
            alpha_grad=0.0,
            alpha_spectral=config["train"].get("alpha_spectral", 0.1),
            alpha_station=config["train"].get("alpha_station", 0.1),
            ghap_mean=config["data"].get("ghap_mean", 15.0),
            ghap_std=config["data"].get("ghap_std", 20.0),
            underestimate_penalty=config["train"].get("underestimate_penalty", 1.0),
            ffl_alpha=config["train"].get("ffl_alpha", 1.0),
        )
        self.ghap_mean = config["data"].get("ghap_mean", 15.0)
        self.ghap_std  = config["data"].get("ghap_std",  20.0)

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
        if torch.isnan(pred).any():
            return sum(p.sum() * 0.0 for p in self.model.parameters())
        loss, metrics = self.criterion(
            pred, batch["target"],
            station_pixels=batch.get("station_pixels"),
            station_values=batch.get("station_values"),
            station_count=batch.get("station_count"),
        )
        if torch.isnan(loss):
            return sum(p.sum() * 0.0 for p in self.model.parameters())
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss, _ = self.criterion(pred, batch["target"])
        rmse = self._rmse(pred, batch["target"])
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/rmse", rmse, prog_bar=True, sync_dist=True)
        return loss

    def _rmse(self, pred, target):
        pf = torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0)
        tf = target.float()
        po = pf * self.ghap_std + self.ghap_mean
        to = tf * self.ghap_std + self.ghap_mean
        mask = (to > 0) & torch.isfinite(to) & torch.isfinite(po)
        d = mask.float().sum().clamp(min=1.0)
        mse = ((po - to)**2 * mask.float()).sum() / d
        return torch.sqrt(mse)

    def configure_optimizers(self):
        tc = self.config["train"]
        lr = tc.get("learning_rate", 5e-5)
        wd = tc.get("weight_decay", 0.05)
        epochs = tc.get("epochs", 25)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=tc.get("min_lr", 1e-6))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
