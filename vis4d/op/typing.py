"""Vis4D op typing."""
from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class Proposals(NamedTuple):
    """Output structure for 2D bounding box proposals."""

    boxes: list[Tensor]
    scores: list[Tensor]
