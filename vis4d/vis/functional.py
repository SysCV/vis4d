"""Function interface for visualization functions."""
from __future__ import annotations

from vis4d.common.typing import (
    NDArrayF64,
    NDArrayI64,
    NDArrayNumber,
    NDArrayUI8,
)
from vis4d.vis.image.base import CanvasBackend, ImageViewerBackend
from vis4d.vis.image.bounding_box_visualizer import BoundingBoxVisualizer
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.vis.image.utils import preprocess_boxes, preprocess_image
from vis4d.vis.image.viewer import MatplotlibImageViewer
from vis4d.vis.pointcloud.pointcloud_visualizer import PointCloudVisualizer
from vis4d.vis.util import generate_color_map

# ======================== Image ==================================


def show_image(
    image: NDArrayUI8,
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
) -> None:
    """Shows a single image.

    Args:
        image (NDArrayNumber): The image to show
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
    """
    image_viewer.show_images([image])


def draw_bboxes(
    image: NDArrayNumber,
    boxes: NDArrayF64,
    scores: None | NDArrayF64 = None,
    class_ids: None | NDArrayI64 = None,
    track_ids: None | NDArrayI64 = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    canvas: CanvasBackend = PillowCanvasBackend(),
) -> NDArrayUI8:
    """Draws the predicted bounding boxes into the given image.

    Args:
        image (NDArrayNumber): The image to draw the bboxes into.
        boxes (NDArrayF64): Predicted bounding boxes.
        scores (None | NDArrayF64, optional): Predicted scores.
            Defaults to None.
        class_ids (NDArrayI64, optional): Predicted class ids.
            Defaults to None.
        track_ids (NDArrayI64, optional): Predicted track ids.
            Defaults to None.
        class_id_mapping (dict[int, str], optional): Mapping from class id to
            name. Defaults to None.
        n_colors (int, optional): Number of colors to use for color palette.
            Defaults to 50.
        image_mode (str, optional): Image Mode.. Defaults to "RGB".
        canvas (CanvasBackend, optional): Canvas backend to use.
            Defaults to PillowCanvasBackend().

    Returns:
        NDArrayUI8: The image with boxes drawn into it,
    """
    image = preprocess_image(image, image_mode)
    box_data = preprocess_boxes(
        boxes,
        scores,
        class_ids,
        track_ids,
        color_palette=generate_color_map(n_colors),
        class_id_mapping=class_id_mapping,
    )
    canvas.create_canvas(image)

    for corners, label, color in zip(*box_data):
        canvas.draw_box(corners, color)
        canvas.draw_text((corners[0], corners[1]), label)
    return canvas.as_numpy_image()


def imshow_bboxes(
    image: NDArrayNumber,
    boxes: NDArrayF64,
    scores: None | NDArrayF64 = None,
    class_ids: None | NDArrayI64 = None,
    track_ids: None | NDArrayI64 = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
) -> None:
    """Shows the bounding boxes overlayed on the given image.

    Args:
        image (NDArrayNumber): Background Image
        boxes (NDArrayF64): Boxes to show. Shape [N, 4] with
                                            (x1,y1,x2,y2) as corner convention
        scores (NDArrayF64, optional): Score for each box shape [N]
        class_ids (NDArrayI64, optional): Class id for each box shape [N]
        track_ids (NDArrayI64, optional): Track id for each box shape [N]
        class_id_mapping (dict[int, str], optional): Mapping to convert
                                                    class id to class name
        n_colors (int, optional): Number of distinct colors used to color the
                                  boxes. Defaults to 50.
        image_mode (str, optional): Image channel mode (RGB or BGR).
    """
    img = draw_bboxes(
        image,
        boxes,
        scores,
        class_ids,
        track_ids,
        class_id_mapping,
        n_colors,
        image_mode,
    )
    show_image(img)


# =========================== Pointcloud ===================================
def show_points(
    points_xyz: NDArrayF64,
    semantics: NDArrayI64 | None = None,
    instances: NDArrayI64 | None = None,
    colors: NDArrayF64 | None = None,
    backend: str = "open3d",
) -> None:
    """Visualizes point cloud data.

    Args:
        points_xyz: xyz coordinates of the points shape [B, N, 3]
        semantics: semantic ids of the points shape [B, N, 1]
        instances: instance ids of the points shape [B, N, 1]
        colors: colors of the points shape [B, N,3] and ranging from  [0,1]
        backend (str): Which visualization backend to use. Choice from [open3d]
    """
    vis = PointCloudVisualizer(backend)
    vis.process_single(points_xyz, semantics, instances, colors)
    vis.show()
