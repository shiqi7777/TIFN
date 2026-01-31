import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pytorch_lightning as pl
import torch
from TIFN.datamodule.dataset import InkmlDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

from .vocab import vocab


MAX_SIZE = 32e4  # change here accroading to your GPU memory

@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    img_mask: LongTensor  # [b, H, W]
    traj: FloatTensor  # [B, T, C]    (padding 到同一长度)
    traj_mask: LongTensor  # [B, T]       (1=valid, 0=pad)
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            img_mask=self.img_mask.to(device),
            traj=self.traj.to(device),
            traj_mask=self.traj_mask.to(device),
            indices=self.indices,
        )

def collate_fn(batch: List[Tuple[str, torch.Tensor, List[str], torch.Tensor, torch.Tensor]]):
    fnames = [b[0] for b in batch]
    images = [b[1] for b in batch]
    targets = [vocab.words2indices(b[2]) for b in batch]

    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    max_h, max_w = max(heights), max(widths)
    B = len(images)
    C_img = images[0].shape[0]  # 从第一个样本获取通道数

    x = torch.zeros(B, C_img, max_h, max_w, dtype=images[0].dtype)
    x_mask = torch.ones(B, max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        x[i, :, :h, :w] = img
        x_mask[i, :h, :w] = False  # False=valid


    trajs = [b[3] for b in batch]
    # trajs_tc = [_to_tc(s) for s in trajs_raw]  # 每个 [T_i, C]

    T_max = max(s.shape[0] for s in trajs)
    C_traj = trajs[0].shape[1]

    traj = torch.zeros(B, T_max, C_traj, dtype=torch.float32)
    traj_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, s in enumerate(trajs):
        t_i = s.shape[0]
        traj[i, :t_i] = s
        traj_mask[i, :t_i] = False  # 有效部分标记为 False=valid

    return Batch(fnames, x, x_mask, traj, traj_mask, targets)



class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        annotation_dir: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../crohme2019",
        test_year: str = "2019",
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = True,
        img_size = (768, 256)
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.annotation_dir = annotation_dir
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug
        self.img_size = img_size
        self.use_sig_feat = True

        print(f"Load data from: {self.annotation_dir}")

    def setup(self, stage: Optional[str] = None) -> None:

        train_anno = os.path.join(self.annotation_dir, "crohme2019_train.txt")
        test_anno = os.path.join(self.annotation_dir, f"crohme{self.test_year}_test.txt")

        train_root = os.path.join(self.annotation_dir, "train")
        test_root = os.path.join(self.annotation_dir, self.test_year)

        if stage == "fit" or stage is None:
            self.train_dataset = InkmlDataset(
                annotation_path=train_anno,
                root_dir=train_root,
                is_train=True,
                scale_aug=self.scale_aug,
                img_size=self.img_size,
                use_sig_feat=self.use_sig_feat,
            )
            self.val_dataset = InkmlDataset(
                annotation_path=test_anno,
                root_dir=test_root,
                is_train=False,
                scale_aug=False,
                img_size=self.img_size,
                use_sig_feat=self.use_sig_feat,
            )

        if stage == "test" or stage is None:
            self.test_dataset = InkmlDataset(
                annotation_path=test_anno,
                root_dir=test_root,
                is_train=False,
                scale_aug=False,
                img_size=self.img_size,
                use_sig_feat=self.use_sig_feat,
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=False
        )
