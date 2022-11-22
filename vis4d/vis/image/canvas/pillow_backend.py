"""Pillow backend implementation to draw on images."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vis4d.vis.image.base import CanvasBackend, NDArrayUI8


class PillowCanvasBackend(CanvasBackend):
    """Canvas backend using Pillow."""

    def __init__(self, font: ImageFont.Font | None = None) -> None:
        """Creates a new canvas backend.

        Args:
            font (ImageFont): Pillow font to use for the label.
        """
        self._image_draw: ImageDraw.ImageDraw | None = None
        self._font = font if font is not None else ImageFont.load_default()
        self._image: Image.Image | None = None

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

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if image is None and image_hw is None:
            raise ValueError("Image or Image Shapes required to create canvas")
        if image_hw is not None:
            image = np.zeros(image_hw)
        self._image = Image.fromarray(image)
        self._image_draw = ImageDraw.Draw(self._image)

    def draw_box(
        self,
        corners: tuple[float, float, float, float],
        label: str,
        color: tuple[float, ...],
    ):
        """Draws a box onto the given canvas.

        Args:
            corners (list[float]): Containing [x1,y2,x2,y2] the corners of
                                    the box
            label (str): Label of the box.
            color (tuple(float)): Color of the box [0,255]

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )

        self._image_draw.rectangle(corners, outline=color)
        self._image_draw.text(
            corners[:2], label, (255, 255, 255), font=self._font
        )

    def as_numpy_image(self) -> NDArrayUI8:
        """Returns the current canvas as numpy image.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image is None:
            raise ValueError(
                "No Image initialized! Did you call 'create_canvas'?"
            )
        return np.asarray(self._image)

    def save_to_disk(self, image_path: str):
        """Writes the current canvas to disk.

        Args:
            image_path (str): Full image path (with file name and ending).

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image is None:
            raise ValueError(
                "No Image initialized! Did you call 'create_canvas'?"
            )
        self._image.save(image_path)
