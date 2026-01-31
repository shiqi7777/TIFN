import math

from typing import Tuple, Any, Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from iisignature import siglength, logsiglength
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc


# DenseNet-B
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super(DenseNet, self).__init__()
        n_dense_blocks = num_layers
        n_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(
            1, n_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.c2_channels = n_channels


        self.dense1 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)
        self.c3_channels = n_out_channels


        n_channels = n_out_channels
        self.dense2 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout
        )

        self.out_channels = n_channels + n_dense_blocks * growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask, return_pyramid: bool = False):
        # x: [B, 1, H', W'], x_mask: [B, H', W'] (bool/long, True=pad)
        out = self.conv1(x) # [B, 2*gr, H'/2,  W'/2]
        out = self.norm1(out)   # [B, 2*gr, H'/2,  W'/2]
        out_mask = x_mask[:, 0::2, 0::2]    # [B, H'/2,  W'/2]
        out = F.relu(out, inplace=True)

        out = F.max_pool2d(out, 2, ceil_mode=True)  # [B, 2*gr, H'/4,  W'/4]
        out_mask = out_mask[:, 0::2, 0::2]  # [B, H'/4,  W'/4]
        I2, M2 = out, out_mask  # I2/M2: 1/4

        out = self.dense1(out)  # [B, C?, H'/4,  W'/4]
        out = self.trans1(out)  # [B, C3, H'/8,  W'/8]
        out_mask = out_mask[:, 0::2, 0::2]  # [B, H'/8,  W'/8]
        I3, M3 = out, out_mask  # I3/M3: 1/8

        out = self.dense2(out)  # [B, C?, H'/8,  W'/8]
        out = self.trans2(out)  # [B, C4, H'/16, W'/16]
        out_mask = out_mask[:, 0::2, 0::2]  # [B, H'/16, W'/16]

        out = self.dense3(out)  # [B, C4', H'/16, W'/16]
        out = self.post_norm(out)   # [B, C4', H'/16, W'/16]

        I4, M4 = out, out_mask  # I4/M4: 1/16
        if return_pyramid:
            return (I2, I3, I4), (M2, M3, M4)
        else:
            return out, out_mask


def _to_tokens(feat, mask):
    B, D, H, W = feat.shape
    q = feat.permute(0, 2, 3, 1).reshape(B, H * W, D)
    m = mask.reshape(B, H * W)
    return q, m, H, W

class TrajPyramid(nn.Module):
    """
    输出:
      {
        'S2': {'feat':[B,T/2,D], 'mask':[B,T/2]},
        'S3': {'feat':[B,T/4,D], 'mask':[B,T/4]},
        'S4': {'feat':[B,T/8,D], 'mask':[B,T/8]},
      }
    """
    def __init__(self, d_model=512, nhead=8, in_dim=6):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=1, stride=2)
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1, stride=2)
        self.conv4 = nn.Conv1d(d_model, d_model, 3, padding=1, stride=2)

    @staticmethod
    def _downsample_mask(mask, times):
        B, T = mask.shape
        m = mask
        for _ in range(times):
            if m.shape[1] % 2 == 1:
                m = torch.cat([m, m[:, -1:]], dim=1)
            m = m.view(B, -1, 2).all(dim=2)
        return m

    def forward(self, x, x_pad_mask, *args, **kwargs):
        x = self.in_proj(x)                 # [B,T,D]
        x_c = x.transpose(1, 2)             # [B,D,T]

        s2 = F.gelu(self.conv2(x_c))        # [B,D,T/2]
        s3 = F.gelu(self.conv3(s2))         # [B,D,T/4]
        s4 = F.gelu(self.conv4(s3))         # [B,D,T/8]

        m2 = self._downsample_mask(x_pad_mask, 1)
        m3 = self._downsample_mask(x_pad_mask, 2)
        m4 = self._downsample_mask(x_pad_mask, 3)

        return {
            'S2': {'feat': s2.transpose(1, 2), 'mask': m2},
            'S3': {'feat': s3.transpose(1, 2), 'mask': m3},
            'S4': {'feat': s4.transpose(1, 2), 'mask': m4},
        }


class CrossAttn2D(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, Q, Qmask, K, Kmask):
        """
        Q: [B, Nq, D],  Qmask: [B, Nq] (True=pad) —— 仅用于输出置零
        K: [B, Nk, D],  Kmask: [B, Nk] (True=pad)
        """
        out, _ = self.mha(Q, K, K, key_padding_mask=Kmask)  # [B, Nq, D]
        out = self.ln(out + Q)                              # [B, Nq, D]
        if Qmask is not None:
            out = out.masked_fill(Qmask.unsqueeze(-1), 0.0) # [B, Nq, D]    将 out 张量中 Qmask 为 True 的位置替换为 0.0
        return out

class PAMF(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        # 对每个尺度的 D 维先投成 1 维分数，然后在尺度维做 softmax
        self.score = nn.Linear(d_model, 1)

    def forward(self, M2, M3, M4):
        """
        输入: 三个尺度已对齐到同空间大小的二维特征
        M*: [B, H, W, D]
        返回: Mo [B, H, W, D]
        """
        stack = torch.stack([M2, M3, M4], dim=3)            # [B, H, W, 3, D]
        score = self.score(stack).squeeze(-1)               # [B, H, W, 3]
        a = torch.softmax(score, dim=-1).unsqueeze(-1)      # [B, H, W, 3, 1]
        Mo = (a * stack).sum(dim=3)                         # [B, H, W, D]
        return Mo

def _to_tokens(feat_4d: torch.Tensor, mask_2d: torch.Tensor):
    # feat_4d: [B, D, H, W] 或 [B, H, W, D] 均可先统一
    if feat_4d.dim() == 4 and feat_4d.shape[1] != feat_4d.shape[-1]:
        # [B, D, H, W] -> [B, H*W, D]
        B, D, H, W = feat_4d.shape
        q = feat_4d.permute(0, 2, 3, 1).reshape(B, H * W, D)
    else:
        B, H, W, D = feat_4d.shape
        q = feat_4d.reshape(B, H * W, D)
    m = mask_2d.reshape(B, H * W)  # True=pad
    return q, m, H, W

class AdaptiveBlock(nn.Module):
    """
    Trajectory-conditioned channel re-weighting
    """
    def __init__(self, d_model=512, hidden_ratio=0.25):
        super().__init__()
        hidden = int(d_model * hidden_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Sigmoid()
        )

    def forward(self, img_feat, traj_feat, traj_mask):
        """
        img_feat : [B, H, W, D]
        traj_feat: [B, T, D]
        traj_mask: [B, T] (True=pad)
        """
        # 1) trajectory global pooling (masked mean)
        mask = (~traj_mask).float().unsqueeze(-1)  # [B,T,1]
        pooled = (traj_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # [B,D]

        # 2) channel gate
        gate = self.mlp(pooled).unsqueeze(1).unsqueeze(1)  # [B,1,1,D]

        # 3) channel-wise modulation (residual-safe)
        return img_feat * gate + img_feat


class MultiScaleBimodal2D(nn.Module):
    """
    输入:
      - P2,P3,P4: 图像侧三个尺度 (已经 1x1 投到 d_model)
      - M2,M3,M4: 对应 mask (True=pad)
      - traj_pyr:  TrajPyramid 的输出
    输出:
      - Mo: [B, H4, W4, D]  (融合后的 /16 特征)
      - M4: [B, H4, W4]     (沿用图像 /16 的 mask)
    """
    def __init__(self, d_model=512, nhead=8, dropout=0.1, use_multiscale=True):
        super().__init__()
        self.cross2 = CrossAttn2D(d_model, nhead, dropout)
        self.cross3 = CrossAttn2D(d_model, nhead, dropout)
        self.cross4 = CrossAttn2D(d_model, nhead, dropout)

        # Adaptive Blocks（每个尺度一个）
        self.adapt2 = AdaptiveBlock(d_model)
        self.adapt3 = AdaptiveBlock(d_model)
        self.adapt4 = AdaptiveBlock(d_model)

        self.pamf = PAMF(d_model)
        self.use_multiscale = use_multiscale

    def forward(self, P2, M2, P3, M3, P4, M4, traj_pyr):
        # P2:[B,D,H/4,W/4], P3:[B,D,H/8,W/8], P4:[B,D,H/16,W/16]
        # M2:[B,H/4,W/4],   M3:[B,H/8,W/8],   M4:[B,H/16,W/16]

        B, D, H4, W4 = P4.shape
        S2, S3, S4 = traj_pyr['S2'], traj_pyr['S3'], traj_pyr['S4']
        size_16 = (H4, W4)

        if self.use_multiscale:
            # 对齐到 /16 的空间大小
            P2s = F.adaptive_avg_pool2d(P2, size_16)  # [B, D, H/16, W/16]
            P3s = F.adaptive_avg_pool2d(P3, size_16)  # [B, D, H/16, W/16]

            M2s = F.interpolate(M2.unsqueeze(1).float(), size=size_16, mode='nearest').squeeze(1).bool()  # [B,H/16,W/16]
            M3s = F.interpolate(M3.unsqueeze(1).float(), size=size_16, mode='nearest').squeeze(1).bool()  # [B,H/16,W/16]

            # token 化（行优先展平）
            Q2, Qm2, _, _ = _to_tokens(P2s, M2s)  # Q2:[B, H4*W4, D], Qm2:[B,H4*W4]
            Q3, Qm3, _, _ = _to_tokens(P3s, M3s)  # Q3:[B, H4*W4, D]
            Q4, Qm4, _, _ = _to_tokens(P4, M4)  # Q4:[B, H4*W4, D]


            # 二维跨模态注意力（Image→Q, Seq→K/V）
            M2_tok = self.cross2(Q2, Qm2, S2['feat'], S2['mask']).view(B, H4, W4, D)  # [B, H/16, W/16, D]
            M3_tok = self.cross3(Q3, Qm3, S3['feat'], S3['mask']).view(B, H4, W4, D)  # [B, H/16, W/16, D]
            M4_tok = self.cross4(Q4, Qm4, S4['feat'], S4['mask']).view(B, H4, W4, D)  # [B, H/16, W/16, D]

            # ---- Adaptive Block (Trajectory → Image channel modulation) ----
            M2_tok = self.adapt2(M2_tok, S2['feat'], S2['mask'])
            M3_tok = self.adapt3(M3_tok, S3['feat'], S3['mask'])
            M4_tok = self.adapt4(M4_tok, S4['feat'], S4['mask'])

            # 位置感知多尺度融合
            Mo = self.pamf(M2_tok, M3_tok, M4_tok)  # [B, H/16, W/16, D]

        else:
            # 只用最深层 P4-S4 进行跨模态注意力
            Q4, Qm4, _, _ = _to_tokens(P4, M4)
            M4_tok = self.cross4(Q4, Qm4, S4['feat'], S4['mask']).view(B, H4, W4, D)
            M4_tok = self.adapt4(M4_tok, S4['feat'], S4['mask'])
            Mo = M4_tok  # 无 PAMF，直接输出

        return Mo, M4

class ImageEncoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()

        self.model = DenseNet(growth_rate=growth_rate, num_layers=num_layers)

        self.feature_proj = nn.Conv2d(self.model.out_channels, d_model, kernel_size=1)


        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        self.fpn_lat2 = nn.Conv2d(self.model.c2_channels, d_model, kernel_size=1)
        self.fpn_lat3 = nn.Conv2d(self.model.c3_channels, d_model, kernel_size=1)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: FloatTensor, img_mask: LongTensor):
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # 提取多尺度特征
        (I2, I3, I4), (M2, M3, M4) = self.model(img, img_mask, return_pyramid=True)
        # I2:[B,C2,H/4,W/4], I3:[B,C3,H/8,W/8], I4:[B,C4,H/16,W/16]
        # M2:[B,H/4,W/4], M3:[B,H/8,W/8], M4:[B,H/16,W/16]

        # 1x1 投到 d_model
        P2 = self.fpn_lat2(I2)  # [B, D, H/4,  W/4]
        P3 = self.fpn_lat3(I3)  # [B, D, H/8,  W/8]
        P4 = self.feature_proj(I4)  # [B, D, H/16, W/16]

        # 位置编码仅在最终 memory 上再加；P2/P3/P4 原样给融合头
        return (P2, P3, P4), (M2, M3, M4)

class Encoder(pl.LightningModule):
    """
    用法：
        enc = FusionEncoder(d_model=512, growth_rate=32, num_layers=6, nhead=8)
        memory, memory_mask = enc(img, img_mask, traj, traj_mask)
        -> memory: [B, H/16, W/16, D], memory_mask: [B, H/16, W/16]
    """
    def __init__(self, d_model: int, growth_rate: int, num_layers: int,
                 nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.img_enc = ImageEncoder(d_model, growth_rate, num_layers)
        D = logsiglength(2, 2)
        # D = siglength(2, 4)
        self.traj_pyr = TrajPyramid(d_model=d_model, nhead=nhead, in_dim=D)
        # self.traj_pyr = TrajPyramid(d_model=d_model)
        self.fusion = MultiScaleBimodal2D(d_model=d_model, nhead=nhead, dropout=dropout, use_multiscale = True)
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    @torch.no_grad()
    def _target_lengths_from_pyramid(self, P2, P3, P4):

        H4, W4 = P4.shape[-2], P4.shape[-1]
        L4 = H4 * W4
        # 三个尺度的序列侧都池到 /16 的 token 数
        return {'L2': L4, 'L3': L4, 'L4': L4}

    def forward(
        self,
        img: FloatTensor,          # [B, 1, H', W']
        img_mask: LongTensor,      # [B, H', W']    True=pad
        traj: FloatTensor,         # [B, T, 6]
        traj_pad_mask: LongTensor, # [B, T]         True=pad
    ) -> Tuple[FloatTensor, LongTensor]:
        # 1) 图像侧多尺度
        (P2, P3, P4), (M2, M3, M4) = self.img_enc(img, img_mask)
        H4, W4 = P4.shape[-2], P4.shape[-1]
        # P2:[B,D,H/4,W/4], P3:[B,D,H/8,W/8], P4:[B,D,H/16,W/16]

        # 2) 轨迹侧金字塔，长度与三层图像 token 数对齐
        target_lengths = self._target_lengths_from_pyramid(P2, P3, P4)

        traj_pad_mask = traj_pad_mask.bool()    # [B,T]
        all_zero = (traj.abs().sum(dim=-1) < 1e-8)  # [B,T]  True=全0
        traj_pad_mask = traj_pad_mask | all_zero  # 把全0也当成pad

        traj_pyr = self.traj_pyr(traj, traj_pad_mask, target_lengths, grid_shape=(H4, W4))   # S2/S3/S4: {'feat':[B,N,D], 'mask':[B,N]}，N=H/16*W/16

        # 3) 跨模态融合（Image→Q, Seq→K/V）+ PAMF 到 /16
        Mo, mem_mask = self.fusion(P2, M2, P3, M3, P4, M4, traj_pyr)   # Mo: [B, H/16, W/16, D], mem_mask: [B, H/16, W/16]

        # 4) 位置编码 + LN（作为最终 memory）
        memory = self.pos_enc_2d(Mo, mem_mask)                        # [B,H/16,W/16,D]
        memory = self.norm(memory)                                    # [B,H/16,W/16,D]
        return memory, mem_mask
