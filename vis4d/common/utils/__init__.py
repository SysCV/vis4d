"""Utility code."""

from .distributed import get_rank, get_world_size
from .time import Timer, timeit

__all__ = ["timeit", "Timer", "get_rank", "get_world_size"]
