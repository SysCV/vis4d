"""data utils."""
import math
import sys
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from detectron2.structures import Instances
from PIL import Image

from openmt.struct import Boxes2D


def im_decode(im_bytes: bytes) -> np.ndarray:
    """Decode to image (numpy array, BGR) from bytes."""
    pil_img = Image.open(BytesIO(bytearray(im_bytes)))
    np_img = np.array(pil_img)[..., [2, 1, 0]]  # type: np.ndarray
    return np_img


def str_decode(str_bytes: bytes, encoding: Optional[str] = None) -> str:
    """Decode to string from bytes."""
    if encoding is None:
        encoding = sys.getdefaultencoding()
    return str_bytes.decode(encoding)


def target_to_box2d(target: Instances, score_as_logit: bool = True) -> Boxes2D:
    """Convert d2 Instances representing targets to Boxes2D."""
    boxes, cls = (
        target.gt_boxes.tensor,
        target.gt_classes,
    )
    track_ids = target.track_ids if "track_ids" in target._fields else None
    score = torch.ones((boxes.shape[0], 1), device=boxes.device)
    if score_as_logit:
        score *= math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    return Boxes2D(torch.cat([boxes, score], -1), cls, track_ids)
