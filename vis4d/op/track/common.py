"""Common classes and functions for tracking"""
from typing import NamedTuple

from torch import Tensor

class TrackOut(NamedTuple):
    """Output of track model."""

    boxes: list[Tensor]  # (N, 4)
    class_ids: list[Tensor]
    scores: list[Tensor]
    track_ids: list[Tensor]
