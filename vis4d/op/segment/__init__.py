"""Segmentor module."""
from .base import BaseSegmentor
from .mmseg import MMEncDecSegmentor

__all__ = ["BaseSegmentor", "MMEncDecSegmentor"]
