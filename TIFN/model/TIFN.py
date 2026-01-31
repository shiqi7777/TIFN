from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from TIFN.utils.utils import Hypothesis

from .decoder import Decoder
from .encoder import Encoder


class TIFN(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        seq_c_in: int = 6,
        nhead_encoder: int = 8
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead_encoder,
        )

        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self._expect_traj_c_in = 6

    def forward(
            self,
            img: FloatTensor,  # [b, 1, H', W']
            img_mask: LongTensor,  # [b, H', W']
            traj: FloatTensor,  # [b, L, C_in]
            traj_mask: LongTensor,  # [b, L]
            tgt: LongTensor,  # [2b, l]
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
        # 1) 编码（注意：encoder 已返回 /16 网格）
        src, src_mask = self.encoder(img, img_mask, traj, traj_mask)
        #    src       : [b, h16, w16, d]
        #    src_mask  : [b, h16, w16]

        # 2) 为双向解码 duplicate（保持旧逻辑）
        src = torch.cat((src, src), dim=0)  # [2b, h16, w16, d]
        src_mask = torch.cat((src_mask, src_mask), 0)  # [2b, h16, w16]

        # 3) 解码
        out = self.decoder(src, src_mask, tgt)  # [2b, l, vocab_size]
        return out

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        traj: FloatTensor,
        traj_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """

        src, src_mask = self.encoder(img, img_mask, traj, traj_mask)
        return self.decoder.beam_search(
            [src], [src_mask],
            beam_size, max_len, alpha, early_stopping, temperature
        )


