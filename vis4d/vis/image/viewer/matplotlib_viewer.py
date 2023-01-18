"""Matplotlib based image viewer."""
from __future__ import annotations

import matplotlib.pyplot as plt

from vis4d.common.typing import NDArrayUI8
from vis4d.vis.image.base import ImageViewerBackend


class MatplotlibImageViewer(ImageViewerBackend):
    """A image viewer using matplotlib.pyplot."""

    def show_images(
        self, images: list[NDArrayUI8], blocking: bool = True
    ) -> None:
        """Shows a list of images.

        Args:
            images (list[NDArrayUI8]): Images to display
            blocking (bool): If the viewer should be blocking and wait
                            for human input after each image.
        """
        for image in images:
            plt.imshow(image)
            plt.axis("off")
            plt.show(block=blocking)
