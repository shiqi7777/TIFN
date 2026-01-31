import math
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn


class WordPosEnc(pl.LightningModule):
    def __init__(
        self, d_model: int = 512, max_len: int = 500, temperature: float = 10000.0
    ) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float)
        dim_t = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0 / (temperature ** (dim_t / d_model))

        inv_freq = torch.einsum("i, j -> i j", position, div_term)

        pe[:, 0::2] = inv_freq.sin()
        pe[:, 1::2] = inv_freq.cos()
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, l, d]

        Returns
        -------
        torch.Tensor
            [b, l, d]
        """
        _, seq_len, _ = x.size()
        emb = self.pe[:seq_len, :]
        x = x + emb[None, :, :]
        return x


class ImgPosEnc(pl.LightningModule):
    """
    This is a more standard lightning_logs of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        d_model: int = 512,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.LongTensor) -> torch.Tensor:
        """add image positional encoding to feature

        Parameters
        ----------
        x : torch.Tensor
            [b, h, w, d]
        mask: torch.LongTensor
            [b, h, w]

        Returns
        -------
        torch.Tensor
            [b, h, w, d]
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            0, self.half_d_model, 2, dtype=torch.float, device=self.device
        )
        inv_feq = 1.0 / (self.temperature ** (dim_t / self.half_d_model))

        pos_x = torch.einsum("b h w, d -> b h w d", x_embed, inv_feq)
        pos_y = torch.einsum("b h w, d -> b h w d", y_embed, inv_feq)

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3)

        x = x + pos
        return x

class SeqPosEnc(nn.Module):
    def __init__(self, d_model, normalize=True):
        super().__init__()
        self.d_model, self.normalize = d_model, normalize
        if normalize:
            self.ln = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x:[b,L,d]
        b, L, d = x.shape
        device = x.device
        pos = torch.arange(L, device=device).float()
        div = torch.exp(torch.arange(0, d, 2, device=device) *
                        -(math.log(10000.0) / d))
        pe = torch.zeros(L, d, device=device)
        pe[:, 0::2] = torch.sin(pos[:, None] * div)
        pe[:, 1::2] = torch.cos(pos[:, None] * div)
        x = x + pe[None]
        return self.ln(x) if self.normalize else x



