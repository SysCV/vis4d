"""Common classes and functions for 3D tracking"""
from typing import NamedTuple

from torch import Tensor


class Track3DOut(NamedTuple):
    """Output of track 3D model."""

    boxes_3d: Tensor
    velocities: Tensor
    class_ids: Tensor
    scores_3d: Tensor
    track_ids: Tensor
