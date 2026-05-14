import torch
import numpy as np


def log_prediction_maps(logger, pred, target, ghap_mean, ghap_std, step, max_samples=4):
    """Log prediction vs ground truth maps to TensorBoard.

    Only called on rank 0 during validation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image
    except ImportError:
        return

    n = min(pred.shape[0], max_samples)

    for i in range(n):
        p = pred[i, 0].detach().float().cpu().numpy() * ghap_std + ghap_mean
        t = target[i, 0].detach().float().cpu().numpy() * ghap_std + ghap_mean

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        vmin = 0
        vmax = max(float(np.nanpercentile(t[t > 0], 95)) if (t > 0).any() else 50, 10)

        im0 = axes[0].imshow(t, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")
        axes[0].set_title("Ground Truth")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(p, vmin=vmin, vmax=vmax, cmap="YlOrRd", aspect="auto")
        axes[1].set_title("Prediction")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        diff = p - t
        vd = max(float(np.nanpercentile(np.abs(diff), 95)), 5)
        im2 = axes[2].imshow(diff, vmin=-vd, vmax=vd, cmap="RdBu_r", aspect="auto")
        axes[2].set_title("Error (pred - truth)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        # Convert to image tensor for TensorBoard
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)[:, :, :3]  # Remove alpha
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        plt.close(fig)

        if hasattr(logger, "experiment"):
            logger.experiment.add_image(
                f"val/prediction_sample_{i}", img_tensor, global_step=step
            )
