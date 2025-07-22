"""Base class of canvas for image based visualization."""

from __future__ import annotations

from vis4d.common.typing import NDArrayBool, NDArrayF32, NDArrayUI8


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
        raise NotImplementedError

    def draw_bitmap(
        self,
        bitmap: NDArrayBool,
        color: tuple[int, int, int],
        top_left_corner: tuple[float, float] = (0, 0),
        alpha: float = 0.5,
    ) -> None:
        """Draws a binary mask onto the given canvas.

        Args:
            bitmap (ndarray): The binary mask to draw
            color (tuple[int, int, int]): Color of the box [0,255].
            top_left_corner (tuple(float, float)): Coordinates of top left
                corner of the bitmap. Defaults to (0, 0).
            alpha (float, optional): Alpha value for transparency of this mask.
                Defaults to 0.5.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

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
            color (ttuple[int, int, int]): Color of the line.
            width (int, optional): Line width. Defaults to 0.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def draw_box(
        self,
        corners: tuple[float, float, float, float],
        color: tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draws a box onto the given canvas.

        Args:
            corners (list[float]): Containing [x1,y1,x2,y2] the corners of
                the box.
            color (tuple[int, int, int]): Color of the box [0,255].
            width (int, optional): Line width. Defaults to 1.

        Raises:
            ValueError: If the canvas is not initialized.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

    def draw_box_3d(
        self,
        corners: list[tuple[float, float, float]],
        color: tuple[int, int, int],
        intrinsics: NDArrayF32,
        width: int = 0,
        camera_near_clip: float = 0.15,
        plot_heading: bool = True,
    ) -> None:
        """Draws a line between two points.

        Args:
            corners (list[tuple[float, float, float]]): Containing the eight
                corners of the box.
            color (tuple[int, int, int]): Color of the line.
            intrinsics (NDArrayF32): Camera intrinsics matrix.
            width (int, optional): The width of the line. Defaults to 0.
            camera_near_clip (float, optional): The near clipping plane of the
                camera. Defaults to 0.15.
            plot_heading (bool, optional): If True, the heading of the box will
                be plotted as a line. Defaults to True.
        """
        raise NotImplementedError

    def as_numpy_image(self) -> NDArrayUI8:
        """Returns the current canvas as numpy image."""
        raise NotImplementedError

    def save_to_disk(self, image_path: str) -> None:
        """Writes the current canvas to disk.

        Args:
            image_path (str): Full image path (with file name and ending).
        """
        raise NotImplementedError
