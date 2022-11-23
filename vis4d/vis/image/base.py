"""Contains base class implementations for image based visualization."""
from __future__ import annotations

from vis4d.common.typing import NDArrayBool, NDArrayUI8


class CanvasBackend:
    """Abstract interface that allows to draw on images.

    Supports drawing different bounding boxes on top of an image.
    """

    def create_canvas(
        self,
        image: NDArrayUI8 | None = None,
        image_hw: tuple[int, int] | None = None,
    ) -> None:
        """Creates a new canvas with a given image or shape internally.

        Either provide a background image or the desired height, width
        of the canvas.

        Args:
            image (np.array[uint8] | None): Numpy array with a background image
            image_hw (tuple[int, int] | None): height, width of the canvas
        """
        raise NotImplementedError()

    def draw_bitmap(
        self,
        bitmap: NDArrayBool,
        color: tuple[float, float, float],
        top_left_corner: tuple[float, float] = (0, 0),
        alpha: float = 0.5,
    ):
        """Draws a binary mask onto the given canvas.

        Args:
            bitmap (ndarray): The binary mask to draw
            color (tuple(float)): Color of the box [0,255]
            top_left_corner (tuple(float, float)): Coordinates of top left
                                                    corner of the bitmap
            alpha (float): Alpha value for transparency of this mask
        """
        raise NotImplementedError()

    def draw_box(
        self,
        corners: tuple[float, ...],
        label: str,
        color: tuple[float, ...],
    ) -> None:
        """Draws a box onto the given canvas.

        Args:
            corners (list[float]): Containing [x1,y2,x2,y2] the corners of
                                    the box
            label (str): Label of the box.
            color (tuple(float)): Color of the box [0,255]
        """

    def as_numpy_image(self) -> NDArrayUI8:
        """Returns the current canvas as numpy image."""
        raise NotImplementedError()

    def save_to_disk(self, image_path: str) -> None:
        """Writes the current canvas to disk.

        Args:
            image_path (str): Full image path (with file name and ending).
        """
        raise NotImplementedError()


class ImageViewerBackend:
    """Abstract interface that allows to show images."""

    def show_images(
        self, images: list[NDArrayUI8], blocking: bool = True
    ) -> None:
        """Shows a list of images.

        Args:
            images (list[NDArrayUI8]): Images to display
            blocking (bool): If the viewer should be blocking and wait for
                            input after each image.
        """
        raise NotImplementedError()
