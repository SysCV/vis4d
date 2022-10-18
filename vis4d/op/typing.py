"""Vis4D op typing."""
from typing import List, NamedTuple

from torch import Tensor


class Proposals(NamedTuple):
    """Output structure for 2D bounding box proposals."""

    boxes: List[Tensor]
    scores: List[Tensor]
