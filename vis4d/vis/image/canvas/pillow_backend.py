"""Pillow backend implementation to draw on images."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageFont import ImageFont

from vis4d.common.typing import NDArrayBool, NDArrayF32, NDArrayF64, NDArrayUI8

from ..util import get_intersection_point, project_point
from .base import CanvasBackend


class PillowCanvasBackend(CanvasBackend):
    """Canvas backend using Pillow."""

    def __init__(self, font: ImageFont | None = None) -> None:
        """Creates a new canvas backend.

        Args:
            font (ImageFont): Pillow font to use for the label.
        """
        self._image_draw: ImageDraw.ImageDraw | None = None
        self._font = font if font is not None else load_default_font()
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
            white_image = np.ones([*image_hw, 3]) * 255
            image = white_image.astype(np.uint8)
        self._image = Image.fromarray(image)
        self._image_draw = ImageDraw.Draw(self._image)

    def draw_bitmap(
        self,
        bitmap: NDArrayBool,
        color: tuple[int, int, int],
        top_left_corner: tuple[float, float] = (0, 0),
        alpha: float = 0.5,
    ) -> None:
        """Draws a binary mask onto the given canvas.

        Args:
            bitmap (ndarray): The binary mask to draw.
            color (tuple[int, int, int]): Color of the box [0,255].
            top_left_corner (tuple(float, float)): Coordinates of top left
                corner of the bitmap.
            alpha (float): Alpha value for transparency of this mask.

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
        self._image_draw.bitmap(
            top_left_corner, bitmap_pil, fill=color  # type: ignore
        )

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
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draws a box onto the given canvas.

        Args:
            corners (list[float]): Containing [x1,y2,x2,y2] the corners of
                the box.
            color (tuple[int, int, int]): Color of the box [0,255].
            width (int, optional): Line width. Defaults to 1.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )

        self._image_draw.rectangle(corners, outline=color, width=width)

    def draw_rotated_box(
        self,
        corners: list[tuple[float, float]],
        color: tuple[int, int, int],
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
            corners (list[tuple[float, float]]): Containing the four corners of
                the box.
            color (tuple[int, int, int]): Color of the box [0,255].
            width (int, optional): Line width. Defaults to 0.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        assert len(corners) == 4, "2D box must consist of 4 corner points."
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )

        self.draw_line(corners[0], corners[1], color, 2 * width)
        self.draw_line(corners[0], corners[2], color, width)
        self.draw_line(corners[1], corners[3], color, width)
        self.draw_line(corners[2], corners[3], color, width)

        center_forward = np.mean(corners[:2], axis=0, dtype=np.float32)
        center = np.mean(corners, axis=0, dtype=np.float32)
        self.draw_line(
            tuple(center.tolist()),
            tuple(center_forward.tolist()),
            color,
            width,
        )

    def draw_line(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        color: tuple[int, int, int],
        width: int = 0,
    ) -> None:
        """Draw a line onto canvas from point 1 to 2.

        Args:
            point1 (tuple[float, float]): Start point (2D pixel coordinates).
            point2 (tuple[float, float]): End point (2D pixel coordinates).
            color (tuple[int, int, int]): Color of the line.
            width (int, optional): Line width. Defaults to 0.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.line((point1, point2), width=width, fill=color)

    def draw_circle(
        self,
        center: tuple[float, float],
        color: tuple[int, int, int],
        radius: int = 2,
    ) -> None:
        """Draw a circle onto canvas.

        Args:
            center (tuple[float, float]): Center of the circle.
            color (tuple[int, int, int]): Color of the circle.
            radius (int, optional): Radius of the circle. Defaults to 2.
        """
        x1 = center[0] - radius
        y1 = center[1] - radius
        x2 = center[0] + radius
        y2 = center[1] + radius
        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)

    def _draw_box_3d_line(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        color: tuple[int, int, int],
        intrinsics: NDArrayF32,
        width: int = 0,
        camera_near_clip: float = 0.15,
    ) -> None:
        """Draws a line between two points.

        Args:
            point1 (tuple[float, float, float]): The first point. The third
                coordinate is the depth.
            point2 (tuple[float, float, float]): The first point. The third
                coordinate is the depth.
            color (tuple[int, int, int]): Color of the line.
            intrinsics (NDArrayF32): Camera intrinsics matrix.
            width (int, optional): The width of the line. Defaults to 0.
            camera_near_clip (float, optional): The near clipping plane of the
                camera. Defaults to 0.15.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        if point1[2] < camera_near_clip and point2[2] < camera_near_clip:
            return

        if point1[2] < camera_near_clip:
            point1 = get_intersection_point(point1, point2, camera_near_clip)
        elif point2[2] < camera_near_clip:
            point2 = get_intersection_point(point1, point2, camera_near_clip)

        pt1 = project_point(point1, intrinsics)
        pt2 = project_point(point2, intrinsics)

        if self._image_draw is None:
            raise ValueError(
                "No Image Draw initialized! Did you call 'create_canvas'?"
            )
        self._image_draw.line((pt1, pt2), width=width, fill=color)

    def draw_box_3d(
        self,
        corners: list[tuple[float, float, float]],
        color: tuple[int, int, int],
        intrinsics: NDArrayF32,
        width: int = 0,
        camera_near_clip: float = 0.15,
    ) -> None:
        """Draws a 3D box onto the given canvas."""
        # Draw Front
        self._draw_box_3d_line(
            corners[0], corners[1], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[1], corners[5], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[5], corners[4], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[4], corners[0], color, intrinsics, width, camera_near_clip
        )

        # Draw Sides
        self._draw_box_3d_line(
            corners[0], corners[2], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[1], corners[3], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[4], corners[6], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[5], corners[7], color, intrinsics, width, camera_near_clip
        )

        # Draw Back
        self._draw_box_3d_line(
            corners[2], corners[3], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[3], corners[7], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[7], corners[6], color, intrinsics, width, camera_near_clip
        )
        self._draw_box_3d_line(
            corners[6], corners[2], color, intrinsics, width, camera_near_clip
        )

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners[:2], axis=0, dtype=np.float32)
        center_bottom = np.mean(corners[:4], axis=0, dtype=np.float32)
        self._draw_box_3d_line(
            tuple(center_bottom.tolist()),
            tuple(center_bottom_forward.tolist()),
            color,
            intrinsics,
            width,
            camera_near_clip,
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


def load_default_font() -> ImageFont:
    """Load a "better than nothing" default font."""
    f = ImageFont()
    f._load_pilfont_data(  # pylint: disable=protected-access
        # courB08
        BytesIO(
            base64.b64decode(
                b"""
UElMZm9udAo7Ozs7OzsxMDsKREFUQQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYAAAAA//8AAQAAAAAAAAABAAEA
BgAAAAH/+gADAAAAAQAAAAMABgAGAAAAAf/6AAT//QADAAAABgADAAYAAAAA//kABQABAAYAAAAL
AAgABgAAAAD/+AAFAAEACwAAABAACQAGAAAAAP/5AAUAAAAQAAAAFQAHAAYAAP////oABQAAABUA
AAAbAAYABgAAAAH/+QAE//wAGwAAAB4AAwAGAAAAAf/5AAQAAQAeAAAAIQAIAAYAAAAB//kABAAB
ACEAAAAkAAgABgAAAAD/+QAE//0AJAAAACgABAAGAAAAAP/6AAX//wAoAAAALQAFAAYAAAAB//8A
BAACAC0AAAAwAAMABgAAAAD//AAF//0AMAAAADUAAQAGAAAAAf//AAMAAAA1AAAANwABAAYAAAAB
//kABQABADcAAAA7AAgABgAAAAD/+QAFAAAAOwAAAEAABwAGAAAAAP/5AAYAAABAAAAARgAHAAYA
AAAA//kABQAAAEYAAABLAAcABgAAAAD/+QAFAAAASwAAAFAABwAGAAAAAP/5AAYAAABQAAAAVgAH
AAYAAAAA//kABQAAAFYAAABbAAcABgAAAAD/+QAFAAAAWwAAAGAABwAGAAAAAP/5AAUAAABgAAAA
ZQAHAAYAAAAA//kABQAAAGUAAABqAAcABgAAAAD/+QAFAAAAagAAAG8ABwAGAAAAAf/8AAMAAABv
AAAAcQAEAAYAAAAA//wAAwACAHEAAAB0AAYABgAAAAD/+gAE//8AdAAAAHgABQAGAAAAAP/7AAT/
/gB4AAAAfAADAAYAAAAB//oABf//AHwAAACAAAUABgAAAAD/+gAFAAAAgAAAAIUABgAGAAAAAP/5
AAYAAQCFAAAAiwAIAAYAAP////oABgAAAIsAAACSAAYABgAA////+gAFAAAAkgAAAJgABgAGAAAA
AP/6AAUAAACYAAAAnQAGAAYAAP////oABQAAAJ0AAACjAAYABgAA////+gAFAAAAowAAAKkABgAG
AAD////6AAUAAACpAAAArwAGAAYAAAAA//oABQAAAK8AAAC0AAYABgAA////+gAGAAAAtAAAALsA
BgAGAAAAAP/6AAQAAAC7AAAAvwAGAAYAAP////oABQAAAL8AAADFAAYABgAA////+gAGAAAAxQAA
AMwABgAGAAD////6AAUAAADMAAAA0gAGAAYAAP////oABQAAANIAAADYAAYABgAA////+gAGAAAA
2AAAAN8ABgAGAAAAAP/6AAUAAADfAAAA5AAGAAYAAP////oABQAAAOQAAADqAAYABgAAAAD/+gAF
AAEA6gAAAO8ABwAGAAD////6AAYAAADvAAAA9gAGAAYAAAAA//oABQAAAPYAAAD7AAYABgAA////
+gAFAAAA+wAAAQEABgAGAAD////6AAYAAAEBAAABCAAGAAYAAP////oABgAAAQgAAAEPAAYABgAA
////+gAGAAABDwAAARYABgAGAAAAAP/6AAYAAAEWAAABHAAGAAYAAP////oABgAAARwAAAEjAAYA
BgAAAAD/+gAFAAABIwAAASgABgAGAAAAAf/5AAQAAQEoAAABKwAIAAYAAAAA//kABAABASsAAAEv
AAgABgAAAAH/+QAEAAEBLwAAATIACAAGAAAAAP/5AAX//AEyAAABNwADAAYAAAAAAAEABgACATcA
AAE9AAEABgAAAAH/+QAE//wBPQAAAUAAAwAGAAAAAP/7AAYAAAFAAAABRgAFAAYAAP////kABQAA
AUYAAAFMAAcABgAAAAD/+wAFAAABTAAAAVEABQAGAAAAAP/5AAYAAAFRAAABVwAHAAYAAAAA//sA
BQAAAVcAAAFcAAUABgAAAAD/+QAFAAABXAAAAWEABwAGAAAAAP/7AAYAAgFhAAABZwAHAAYAAP//
//kABQAAAWcAAAFtAAcABgAAAAD/+QAGAAABbQAAAXMABwAGAAAAAP/5AAQAAgFzAAABdwAJAAYA
AP////kABgAAAXcAAAF+AAcABgAAAAD/+QAGAAABfgAAAYQABwAGAAD////7AAUAAAGEAAABigAF
AAYAAP////sABQAAAYoAAAGQAAUABgAAAAD/+wAFAAABkAAAAZUABQAGAAD////7AAUAAgGVAAAB
mwAHAAYAAAAA//sABgACAZsAAAGhAAcABgAAAAD/+wAGAAABoQAAAacABQAGAAAAAP/7AAYAAAGn
AAABrQAFAAYAAAAA//kABgAAAa0AAAGzAAcABgAA////+wAGAAABswAAAboABQAGAAD////7AAUA
AAG6AAABwAAFAAYAAP////sABgAAAcAAAAHHAAUABgAAAAD/+wAGAAABxwAAAc0ABQAGAAD////7
AAYAAgHNAAAB1AAHAAYAAAAA//sABQAAAdQAAAHZAAUABgAAAAH/+QAFAAEB2QAAAd0ACAAGAAAA
Av/6AAMAAQHdAAAB3gAHAAYAAAAA//kABAABAd4AAAHiAAgABgAAAAD/+wAF//0B4gAAAecAAgAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYAAAAB
//sAAwACAecAAAHpAAcABgAAAAD/+QAFAAEB6QAAAe4ACAAGAAAAAP/5AAYAAAHuAAAB9AAHAAYA
AAAA//oABf//AfQAAAH5AAUABgAAAAD/+QAGAAAB+QAAAf8ABwAGAAAAAv/5AAMAAgH/AAACAAAJ
AAYAAAAA//kABQABAgAAAAIFAAgABgAAAAH/+gAE//sCBQAAAggAAQAGAAAAAP/5AAYAAAIIAAAC
DgAHAAYAAAAB//kABf/+Ag4AAAISAAUABgAA////+wAGAAACEgAAAhkABQAGAAAAAP/7AAX//gIZ
AAACHgADAAYAAAAA//wABf/9Ah4AAAIjAAEABgAAAAD/+QAHAAACIwAAAioABwAGAAAAAP/6AAT/
+wIqAAACLgABAAYAAAAA//kABP/8Ai4AAAIyAAMABgAAAAD/+gAFAAACMgAAAjcABgAGAAAAAf/5
AAT//QI3AAACOgAEAAYAAAAB//kABP/9AjoAAAI9AAQABgAAAAL/+QAE//sCPQAAAj8AAgAGAAD/
///7AAYAAgI/AAACRgAHAAYAAAAA//kABgABAkYAAAJMAAgABgAAAAH//AAD//0CTAAAAk4AAQAG
AAAAAf//AAQAAgJOAAACUQADAAYAAAAB//kABP/9AlEAAAJUAAQABgAAAAH/+QAF//4CVAAAAlgA
BQAGAAD////7AAYAAAJYAAACXwAFAAYAAP////kABgAAAl8AAAJmAAcABgAA////+QAGAAACZgAA
Am0ABwAGAAD////5AAYAAAJtAAACdAAHAAYAAAAA//sABQACAnQAAAJ5AAcABgAA////9wAGAAAC
eQAAAoAACQAGAAD////3AAYAAAKAAAAChwAJAAYAAP////cABgAAAocAAAKOAAkABgAA////9wAG
AAACjgAAApUACQAGAAD////4AAYAAAKVAAACnAAIAAYAAP////cABgAAApwAAAKjAAkABgAA////
+gAGAAACowAAAqoABgAGAAAAAP/6AAUAAgKqAAACrwAIAAYAAP////cABQAAAq8AAAK1AAkABgAA
////9wAFAAACtQAAArsACQAGAAD////3AAUAAAK7AAACwQAJAAYAAP////gABQAAAsEAAALHAAgA
BgAAAAD/9wAEAAACxwAAAssACQAGAAAAAP/3AAQAAALLAAACzwAJAAYAAAAA//cABAAAAs8AAALT
AAkABgAAAAD/+AAEAAAC0wAAAtcACAAGAAD////6AAUAAALXAAAC3QAGAAYAAP////cABgAAAt0A
AALkAAkABgAAAAD/9wAFAAAC5AAAAukACQAGAAAAAP/3AAUAAALpAAAC7gAJAAYAAAAA//cABQAA
Au4AAALzAAkABgAAAAD/9wAFAAAC8wAAAvgACQAGAAAAAP/4AAUAAAL4AAAC/QAIAAYAAAAA//oA
Bf//Av0AAAMCAAUABgAA////+gAGAAADAgAAAwkABgAGAAD////3AAYAAAMJAAADEAAJAAYAAP//
//cABgAAAxAAAAMXAAkABgAA////9wAGAAADFwAAAx4ACQAGAAD////4AAYAAAAAAAoABwASAAYA
AP////cABgAAAAcACgAOABMABgAA////+gAFAAAADgAKABQAEAAGAAD////6AAYAAAAUAAoAGwAQ
AAYAAAAA//gABgAAABsACgAhABIABgAAAAD/+AAGAAAAIQAKACcAEgAGAAAAAP/4AAYAAAAnAAoA
LQASAAYAAAAA//gABgAAAC0ACgAzABIABgAAAAD/+QAGAAAAMwAKADkAEQAGAAAAAP/3AAYAAAA5
AAoAPwATAAYAAP////sABQAAAD8ACgBFAA8ABgAAAAD/+wAFAAIARQAKAEoAEQAGAAAAAP/4AAUA
AABKAAoATwASAAYAAAAA//gABQAAAE8ACgBUABIABgAAAAD/+AAFAAAAVAAKAFkAEgAGAAAAAP/5
AAUAAABZAAoAXgARAAYAAAAA//gABgAAAF4ACgBkABIABgAAAAD/+AAGAAAAZAAKAGoAEgAGAAAA
AP/4AAYAAABqAAoAcAASAAYAAAAA//kABgAAAHAACgB2ABEABgAAAAD/+AAFAAAAdgAKAHsAEgAG
AAD////4AAYAAAB7AAoAggASAAYAAAAA//gABQAAAIIACgCHABIABgAAAAD/+AAFAAAAhwAKAIwA
EgAGAAAAAP/4AAUAAACMAAoAkQASAAYAAAAA//gABQAAAJEACgCWABIABgAAAAD/+QAFAAAAlgAK
AJsAEQAGAAAAAP/6AAX//wCbAAoAoAAPAAYAAAAA//oABQABAKAACgClABEABgAA////+AAGAAAA
pQAKAKwAEgAGAAD////4AAYAAACsAAoAswASAAYAAP////gABgAAALMACgC6ABIABgAA////+QAG
AAAAugAKAMEAEQAGAAD////4AAYAAgDBAAoAyAAUAAYAAP////kABQACAMgACgDOABMABgAA////
+QAGAAIAzgAKANUAEw==
"""
            )
        ),
        Image.open(
            BytesIO(
                base64.b64decode(
                    b"""
iVBORw0KGgoAAAANSUhEUgAAAx4AAAAUAQAAAAArMtZoAAAEwElEQVR4nABlAJr/AHVE4czCI/4u
Mc4b7vuds/xzjz5/3/7u/n9vMe7vnfH/9++vPn/xyf5zhxzjt8GHw8+2d83u8x27199/nxuQ6Od9
M43/5z2I+9n9ZtmDBwMQECDRQw/eQIQohJXxpBCNVE6QCCAAAAD//wBlAJr/AgALyj1t/wINwq0g
LeNZUworuN1cjTPIzrTX6ofHWeo3v336qPzfEwRmBnHTtf95/fglZK5N0PDgfRTslpGBvz7LFc4F
IUXBWQGjQ5MGCx34EDFPwXiY4YbYxavpnhHFrk14CDAAAAD//wBlAJr/AgKqRooH2gAgPeggvUAA
Bu2WfgPoAwzRAABAAAAAAACQgLz/3Uv4Gv+gX7BJgDeeGP6AAAD1NMDzKHD7ANWr3loYbxsAD791
NAADfcoIDyP44K/jv4Y63/Z+t98Ovt+ub4T48LAAAAD//wBlAJr/AuplMlADJAAAAGuAphWpqhMx
in0A/fRvAYBABPgBwBUgABBQ/sYAyv9g0bCHgOLoGAAAAAAAREAAwI7nr0ArYpow7aX8//9LaP/9
SjdavWA8ePHeBIKB//81/83ndznOaXx379wAAAD//wBlAJr/AqDxW+D3AABAAbUh/QMnbQag/gAY
AYDAAACgtgD/gOqAAAB5IA/8AAAk+n9w0AAA8AAAmFRJuPo27ciC0cD5oeW4E7KA/wD3ECMAn2tt
y8PgwH8AfAxFzC0JzeAMtratAsC/ffwAAAD//wBlAJr/BGKAyCAA4AAAAvgeYTAwHd1kmQF5chkG
ABoMIHcL5xVpTfQbUqzlAAAErwAQBgAAEOClA5D9il08AEh/tUzdCBsXkbgACED+woQg8Si9VeqY
lODCn7lmF6NhnAEYgAAA/NMIAAAAAAD//2JgjLZgVGBg5Pv/Tvpc8hwGBjYGJADjHDrAwPzAjv/H
/Wf3PzCwtzcwHmBgYGcwbZz8wHaCAQMDOwMDQ8MCBgYOC3W7mp+f0w+wHOYxO3OG+e376hsMZjk3
AAAAAP//YmCMY2A4wMAIN5e5gQETPD6AZisDAwMDgzSDAAPjByiHcQMDAwMDg1nOze1lByRu5/47
c4859311AYNZzg0AAAAA//9iYGDBYihOIIMuwIjGL39/fwffA8b//xv/P2BPtzzHwCBjUQAAAAD/
/yLFBrIBAAAA//9i1HhcwdhizX7u8NZNzyLbvT97bfrMf/QHI8evOwcSqGUJAAAA//9iYBB81iSw
pEE170Qrg5MIYydHqwdDQRMrAwcVrQAAAAD//2J4x7j9AAMDn8Q/BgYLBoaiAwwMjPdvMDBYM1Tv
oJodAAAAAP//Yqo/83+dxePWlxl3npsel9lvLfPcqlE9725C+acfVLMEAAAA//9i+s9gwCoaaGMR
evta/58PTEWzr21hufPjA8N+qlnBwAAAAAD//2JiWLci5v1+HmFXDqcnULE/MxgYGBj+f6CaJQAA
AAD//2Ji2FrkY3iYpYC5qDeGgeEMAwPDvwQBBoYvcTwOVLMEAAAA//9isDBgkP///0EOg9z35v//
Gc/eeW7BwPj5+QGZhANUswMAAAD//2JgqGBgYGBgqEMXlvhMPUsAAAAA//8iYDd1AAAAAP//AwDR
w7IkEbzhVQAAAABJRU5ErkJggg==
"""
                )
            )
        ),
    )
    return f
