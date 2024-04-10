"""YOLOX decoder for 2D boxes.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import torch
from torch import Tensor


class YOLOXBBoxDecoder:
    """YOLOX BBox decoder."""

    def __call__(self, points: Tensor, offsets: Tensor) -> Tensor:
        """Apply box offsets to points, used by YOLOX.

        Args:
            points (Tensor): Points. Shape (B, N, 4) or (N, 4).
            offsets (Tensor): Offsets. Has shape (B, N, 4) or (N, 4).

        Returns:
            Tensor: Decoded boxes.
        """
        xys = (offsets[..., :2] * points[:, 2:]) + points[:, :2]
        whs = offsets[..., 2:].exp() * points[:, 2:]

        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes
