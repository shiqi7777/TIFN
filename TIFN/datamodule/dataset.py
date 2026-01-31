import numpy as np
import torch
import torchvision.transforms as tr
from PIL import Image
from torch.utils.data.dataset import Dataset
import os

from .preprocess import render_traces_to_image, Inkml, extract_signature_feature
from .transforms import ScaleAugmentation, ScaleToLimitRange

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

class InkmlDataset(Dataset):
    """
        返回：
          fname       : str
          img_tensor  : [1, H, W]  (float, [-1,1])
          seq_feat    : [L, 2]     (float, 已 padding 到 len=SEQ_LEN)
          seq_mask    : [L]        (long, 1=valid,0=pad)
          caption     : List[str]
    """
    def __init__(
        self,
        annotation_path,
        root_dir,
        is_train=False,
        scale_aug=False,
        img_size=(768, 256),
        use_sig_feat=True,
        sig_k: int = 2,  # k=2 → D=6
        window_size: int = 4,
        interp_points: int = 128,
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.use_sig_feat = use_sig_feat
        self.sig_k = sig_k
        self.window_size = window_size
        self.interp_points = interp_points

        with open(annotation_path, "r") as f:
            self.entries = [line.strip().split("\t") for line in f]

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label = self.entries[idx]
        fname = os.path.splitext(os.path.basename(path))[0]

        inkml_path = os.path.join(self.root_dir, path + ".inkml")
        inkml_obj = Inkml(inkml_path)
        traces = inkml_obj.getTraces()

        # 图像渲染
        img_arr = render_traces_to_image(traces, img_size=self.img_size, invert=True)

        # 保存图片
        img = Image.fromarray(img_arr)
        img_tensor = self.transform(img)

        seq_feat = extract_signature_feature(
            traces,
            k=self.sig_k,
            window_size=self.window_size,
            interp_points=self.interp_points,
            norm="minmax",
        )  # [T, D]

        seq_feat = torch.from_numpy(seq_feat).float()
        seq_mask = torch.ones(len(seq_feat), dtype=torch.long)  # 1 = valid
        caption = label.strip().split(" ")

        return fname, img_tensor, caption, seq_feat, seq_mask
