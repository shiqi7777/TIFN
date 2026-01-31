from pytorch_lightning.callbacks import Callback
import torchvision.utils
import torch
from TIFN.datamodule import vocab


class LogPredictionSamples(Callback):
    def __init__(self, max_samples: int = 16, max_batches: int = 4):
        super().__init__()
        self.max_samples = max_samples
        self.max_batches = max_batches

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx >= self.max_batches:
            return


        device = pl_module.device
        epoch = pl_module.current_epoch
        writer = getattr(trainer.logger, "experiment", None)
        if writer is None:
            return

        num_samples = min(self.max_samples, batch.imgs.shape[0])

        # 图像展示
        imgs = batch.imgs[:num_samples].detach().cpu()
        gray_imgs = imgs[:, 0:1, :, :]
        grid = torchvision.utils.make_grid((gray_imgs + 1) / 2.0, nrow=num_samples)
        writer.add_image(f"val/images_batch_{batch_idx}", grid, epoch)

        # Beam Search
        with torch.no_grad():
            hyps = pl_module.approximate_joint_search(
                batch.imgs[:num_samples].to(device),  # img
                batch.img_mask[:num_samples].to(device),  # img_mask
                batch.traj[:num_samples].to(device),  # traj
                batch.traj_mask[:num_samples].to(device)  # traj_mask
            )
        preds = [vocab.indices2label(h.seq) for h in hyps]
        targets = [vocab.indices2label(seq) for seq in batch.indices[:num_samples]]

        # 合并文本
        for i, (target, pred) in enumerate(zip(targets, preds)):
            text = f"**Target:**\n```\n{target}\n```\n**Prediction:**\n```\n{pred}\n```"
            writer.add_text(f"val/sample_b{batch_idx}_{i}", text, epoch)
