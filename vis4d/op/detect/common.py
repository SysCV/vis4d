"""Common classes and functions for detection."""

from typing import NamedTuple

from torch import Tensor


class DetOut(NamedTuple):
    """Output of the detection model.

    boxes (list[Tensor]): 2D bounding boxes of shape [N, 4] in xyxy format.
    scores (list[Tensor]): confidence scores of shape [N,].
    class_ids (list[Tensor]): class ids of shape [N,].
    """

    boxes: list[Tensor]
    scores: list[Tensor]
    class_ids: list[Tensor]
