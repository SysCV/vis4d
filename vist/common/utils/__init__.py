"""Utility code."""

from .distributed import get_rank, get_world_size
from .time import timeit

__all__ = ["timeit", "get_rank", "get_world_size"]
