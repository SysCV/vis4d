"""Common classes and functions for detection."""
from typing import NamedTuple

from torch import Tensor


class DetOut(NamedTuple):
    """Output of the final detections from RCNN."""

    boxes: list[Tensor]  # N, 4
    scores: list[Tensor]
    class_ids: list[Tensor]
