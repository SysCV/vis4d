"""Pillow backend implementation to draw on images."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vis4d.common.typing import NDArrayBool, NDArrayF64, NDArrayUI8
from vis4d.vis.image.base import CanvasBackend


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
            image = np.zeros(image_hw, dtype=np.uint8)
        self._image = Image.fromarray(image)
        self._image_draw = ImageDraw.Draw(self._image)

    def draw_bitmap(
        self,
        bitmap: NDArrayBool,
        color: tuple[float, float, float],
        top_left_corner: tuple[float, float] = (0, 0),
        alpha: float = 0.5,
    ) -> None:
        """Draws a binary mask onto the given canvas.

        Args:
            bitmap (ndarray): The binary mask to draw
            color (tuple(float)): Color of the box [0,255]
            top_left_corner (tuple(float, float)): Coordinates of top left
                 corner of the bitmap.
            alpha (float): Alpha value for transparency of this mask

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        mask = np.squeeze(bitmap)
        assert len(mask.shape) == 2, "Bitmap expected to have shape [h,w]"

        bitmap_with_alpha: NDArrayF64 = np.repeat(
            mask[:, :, None], 4, axis=2
        ).astype(np.float64)
        bitmap_with_alpha[..., -1] = bitmap_with_alpha[..., -1] * alpha * 255
        bitmap_pil = Image.fromarray(
            bitmap_with_alpha.astype(np.uint8), mode="RGBA"
        )
        self._image_draw.bitmap(top_left_corner, bitmap_pil, fill=color)

    def draw_text(
        self,
        position: tuple[float, float],
        text: str,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Draw text onto canvas at given position.

        Args:
            position (tuple[float, float]): x,y position where the text will
                start.
            text (str): Text to be placed at the given location.
            color (tuple[int, int, int], optional): Text color. Defaults to
                (255, 255, 255).

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.text(position, text, color, font=self._font)

    def draw_box(
        self,
        corners: tuple[float, float, float, float],
        color: tuple[float, float, float],
    ) -> None:
        """Draws a box onto the given canvas.

        Args:
            corners (list[float]): Containing [x1,y2,x2,y2] the corners of
                                    the box
            color (tuple(float)): Color of the box [0,255]

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )

        self._image_draw.rectangle(corners, outline=color)

    def draw_rotated_box(
        self,
        corners: tuple[tuple[float, float], ...],
        color: tuple[float, float, float],
        width: int = 0,
    ) -> None:
        """Draws a box onto the given canvas.

        Corner ordering:

        (2) +---------+ (3)
            |         |
            |         |
            |         |
        (0) +---------+ (1)

        Args:
            corners (tuple[tuple[float, float], ...]): Containing the four
                corners of the box.
            color (tuple(float)): Color of the box [0,255].
            width (int, optional): Line width. Defaults to 0.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        assert len(corners) == 4, "2D box must consist of 4 corner points."
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        for i in range(3):
            self.draw_line(corners[i], corners[i + 1], color, width)

    def draw_line(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        color: tuple[float, float, float],
        width: int = 0,
    ) -> None:
        """Draw a line onto canvas from point 1 to 2.

        Args:
            point1 (tuple[float, float]): Start point (2D pixel coordinates).
            point2 (tuple[float, float]): End point (2D pixel coordinates).
            color (tuple[float, float, float]): Color of the line.
            width (int, optional): Line width. Defaults to 0.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.line((point1, point2), width=width, fill=color)

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

    def save_to_disk(self, image_path: str) -> None:
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
