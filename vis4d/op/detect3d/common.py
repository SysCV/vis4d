"""Common classes and functions for 3D detection."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class Detect3DOut(NamedTuple):
    """Output of detect 3D model.

    Attributes:
        boxes_3d (list[Tensor]): List of bounding boxes (B, N, 10).
        velocities (list[Tensor]): List of velocities (B, N, 3).
        class_ids (list[Tensor]): List of class ids (B, N).
        scores_3d (list[Tensor]): List of scores (B, N).
    """

    boxes_3d: list[Tensor]
    velocities: list[Tensor]
    class_ids: list[Tensor]
    scores_3d: list[Tensor]
