"""Viewer implementations to display images."""

from .base import ImageViewerBackend
from .matplotlib_viewer import MatplotlibImageViewer

__all__ = ["ImageViewerBackend", "MatplotlibImageViewer"]
