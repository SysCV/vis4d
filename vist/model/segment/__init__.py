"""Segmentor module."""
from .base import BaseSegmentor
from .mmseg_wrapper import MMEncDecSegmentor

__all__ = ["BaseSegmentor", "MMEncDecSegmentor"]
