from typing import List
import cv2
import numpy as np
import torch
from torch import Tensor
from PIL import Image


class ScaleToLimitRange:
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)

        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= r <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
            return img

        return img


class ScaleAugmentation:
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img) -> np.ndarray:
        if isinstance(img, Image.Image):
            img = np.array(img)

        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img
