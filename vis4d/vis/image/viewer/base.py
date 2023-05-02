"""Base class of image viewer for image based visualization."""
from __future__ import annotations

from vis4d.common.typing import NDArrayUI8


class ImageViewerBackend:
    """Abstract interface that allows to show images."""

    def show_images(
        self, images: list[NDArrayUI8], blocking: bool = True
    ) -> None:
        """Shows a list of images.

        Args:
            images (list[NDArrayUI8]): Images to display.
            blocking (bool, optional): If the viewer should be blocking and
                wait for input after each image. Defaults to True.
        """
        raise NotImplementedError
