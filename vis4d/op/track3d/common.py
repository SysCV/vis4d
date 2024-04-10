"""Common classes and functions for 3D tracking."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class Track3DOut(NamedTuple):
    """Output of track 3D model.

    Attributes:
        boxes_3d (list[Tensor]): List of bounding boxes (B, N, 10).
        velocities (list[Tensor]): List of velocities (B, N, 3).
        class_ids (list[Tensor]): List of class ids (B, N).
        scores_3d (list[Tensor]): List of scores (B, N).
        track_ids (list[Tensor]): List of track ids (B, N).
    """

    boxes_3d: list[Tensor]
    velocities: list[Tensor]
    class_ids: list[Tensor]
    scores_3d: list[Tensor]
    track_ids: list[Tensor]
