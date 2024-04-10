"""Common classes and functions for tracking."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class TrackOut(NamedTuple):
    """Output of track model.

    Attributes:
        boxes (list[Tensor]): List of bounding boxes (B, N, 4).
        class_ids (list[Tensor]): List of class ids (B, N).
        scores (list[Tensor]): List of scores (B, N).
        track_ids (list[Tensor]): List of track ids (B, N).
    """

    boxes: list[Tensor]
    class_ids: list[Tensor]
    scores: list[Tensor]
    track_ids: list[Tensor]
