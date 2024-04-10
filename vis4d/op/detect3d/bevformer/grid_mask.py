"""Grid mask for BEVFormer."""

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn


class GridMask(nn.Module):
    """Grid Mask Layer."""

    def __init__(
        self,
        use_h: bool,
        use_w: bool,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 0,
        prob: float = 1.0,
    ) -> None:
        """Init."""
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        if np.random.rand() > self.prob:
            return x

        device = x.device
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask_img = Image.fromarray(np.uint8(mask))
        mask_img = mask_img.rotate(r)
        mask = np.asarray(mask_img)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask_tensor = torch.from_numpy(mask).to(x.dtype).to(device)
        if self.mode == 1:
            mask_tensor = 1 - mask_tensor
        mask_tensor = mask_tensor.expand_as(x)
        if self.offset:
            offset = (
                torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
                .to(x.dtype)
                .to(device)
            )
            x = x * mask_tensor + offset * (1 - mask_tensor)
        else:
            x = x * mask_tensor

        return x.view(n, c, h, w)
