import zipfile
import os
from typing import List

import editdistance
import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor

from TIFN.datamodule import Batch, vocab
from TIFN.model.TIFN import TIFN
from TIFN.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,
                              to_bi_tgt_out)


class LitTIFN(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
        test_year: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters() #自动保存模型初始化时（即 __init__() 方法中）传入的超参数

        self.tifn_model = TIFN(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, batch: Batch, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.tifn_model(batch.imgs, batch.img_mask, batch.traj, batch.traj_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.img_mask, batch.traj, batch.traj_mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.img_mask, batch.traj, batch.traj_mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return {
            "img_bases": batch.img_bases,  # list[str]
            "pred_ids": [h.seq for h in hyps],  # list[list[int]]
            "gt_ids": batch.indices,  # list[list[int]]
            "pred_text": [vocab.indices2label(h.seq) for h in hyps],  # list[str]
        }

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        year = getattr(self.hparams, "test_year", None)

        if year is None:
            raise ValueError("test_year not found in hparams; pass it in load_from_checkpoint")

        error_tol = getattr(self.hparams, "error_tol", 0)

        out_dir = os.path.join("test_temp", str(year))
        os.makedirs(out_dir, exist_ok=True)
        zip_path = os.path.join(out_dir, "result.zip")
        tsv_path = os.path.join(out_dir, "pred_ids.tsv")
        summary_path = os.path.join(out_dir, f"eval_summary_{year}.txt")

        matched = matched_le1 = matched_le2 = total = 0

        with zipfile.ZipFile(zip_path, "w") as zip_f, open(tsv_path, "w", encoding="utf-8") as tsv, open(summary_path, "w", encoding="utf-8") as summ:
            tsv.write("img\tpred_ids\tgt_ids\tdist\n")
            summ.write("样本ID\t预测(去$)\t标签\t是否正确\n")
            for out in test_outputs:
                img_bases = out["img_bases"]
                pred_ids_batch = out["pred_ids"]
                gt_ids_batch = out["gt_ids"]
                pred_text_batch = out.get("pred_text") or [vocab.indices2label(p) for p in pred_ids_batch]

                for img_base, pred_ids, gt_ids, pred_text in zip(img_bases, pred_ids_batch, gt_ids_batch, pred_text_batch):

                    # 1) 仍写 LaTeX 文本到 result.zip（兼容比赛/外部评测）
                    content = f"%{img_base}\n${pred_text}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)

                    dist = editdistance.eval(pred_ids, gt_ids)

                    # 3) 统计
                    total += 1
                    if dist == 0: matched += 1
                    if dist <= 1: matched_le1 += 1
                    if dist <= 2: matched_le2 += 1

                    # 4) 也把 id 序列落盘，inference.py 直接读它即可
                    tsv.write(f"{img_base}\t{' '.join(map(str, pred_ids))}\t{' '.join(map(str, gt_ids))}\t{dist}\n")
                    gt_text = vocab.indices2label(gt_ids)
                    ok = "✅" if dist <= error_tol else "❌"
                    summ.write(f"{img_base}\t\t{pred_text}\t\t{gt_text}\t\t{ok}\n")

            if total > 0:
                exp = matched / total * 100.0
                exp1 = matched_le1 / total * 100.0
                exp2 = matched_le2 / total * 100.0
                summ.write(
                    f"\nTOTAL={total}\t"
                    f"ExpRate={exp:.2f}% | "
                    f"ExpRate@1={exp1:.2f}% | "
                    f"ExpRate@2={exp2:.2f}%\n"
                )
            else:
                summ.write("\nTOTAL=0\tExpRate=N/A | ExpRate@1=N/A | ExpRate@2=N/A\n")

        print(f"结果已保存到: {os.path.abspath(zip_path)}")
        print(f"ID 序列写入: {os.path.abspath(tsv_path)}")
        print(f"文本总结已写入: {os.path.abspath(summary_path)}")

        if total > 0:
            print(f"ExpRate={matched / total * 100:.2f}% | "
                  f"ExpRate@1={matched_le1 / total * 100:.2f}% | "
                  f"ExpRate@2={matched_le2 / total * 100:.2f}%")

    def approximate_joint_search(
        self, img: FloatTensor, img_mask: LongTensor,traj: FloatTensor,traj_mask: LongTensor
    ) -> List[Hypothesis]:
        return self.tifn_model.beam_search(img, img_mask, traj, traj_mask, **self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
