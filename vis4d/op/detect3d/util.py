"""Filter boxes3d on distance."""
import numpy as np
import torch
from torch import Tensor

from vis4d.common.imports import DETECTRON2_AVAILABLE

if DETECTRON2_AVAILABLE:
    from detectron2.layers.nms import batched_nms_rotated


def bev_3d_nms(  # pragma: no cover
    boxes3d: Tensor,
    scores_3d: Tensor,
    class_ids: Tensor,
    class_agnostic: bool = True,
    iou_threshold: float = 0.1,
) -> Tensor:
    """BEV 3D NMS in world coordinate."""
    center_x = boxes3d[:, 0].unsqueeze(1)
    center_y = boxes3d[:, 1].unsqueeze(1)
    width = boxes3d[:, 4].unsqueeze(1)
    length = boxes3d[:, 5].unsqueeze(1)
    angle = 180.0 / np.pi * boxes3d[:, 8].unsqueeze(1)

    if class_agnostic:
        class_ids = torch.zeros_like(scores_3d, dtype=torch.int64)

    keep_indices = batched_nms_rotated(
        torch.cat([center_x, center_y, width, length, angle], dim=-1),
        scores_3d,
        class_ids,
        iou_threshold,
    )

    return keep_indices
