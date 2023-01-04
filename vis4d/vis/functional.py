"""Function interface for visualization functions."""
from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy, arrays_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    ArrayLikeBool,
    ArrayLikeFloat,
    ArrayLikeInt,
    NDArrayUI8,
)
from vis4d.vis.image.base import CanvasBackend, ImageViewerBackend
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.vis.image.util import (
    preprocess_boxes,
    preprocess_image,
    preprocess_masks,
)
from vis4d.vis.image.viewer import MatplotlibImageViewer
from vis4d.vis.pointcloud.base import PointCloudVisualizerBackend, Scene3D
from vis4d.vis.pointcloud.viewer.open3d_viewer import (
    Open3DVisualizationBackend,
)
from vis4d.vis.util import DEFAULT_COLOR_MAPPING, generate_color_map

# ======================== Image ==================================


def imshow(
    image: ArrayLike,
    image_mode: str = "RGB",
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
) -> None:
    """Shows a single image.

    Args:
        image (NDArrayNumber): The image to show
        image_mode (str, optional): Image Mode. Defaults to "RGB".
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
    """
    image = preprocess_image(image, image_mode)
    image_viewer.show_images([image])


def draw_masks(
    image: ArrayLike,
    masks: ArrayLikeBool,
    class_ids: ArrayLikeInt | None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    canvas: CanvasBackend = PillowCanvasBackend(),
) -> NDArrayUI8:
    """Draws semantic masks into the given image.

    Args:
        image (ArrayLike): The image to draw the bboxes into.
        masks (ArrayLikeBool): The semantic masks with the same shape as the
            image.
        class_ids (ArrayLikeInt, optional): Predicted class ids.
            Defaults to None.
        n_colors (int, optional): Number of colors to use for color palette.
            Defaults to 50.
        image_mode (str, optional): Image Mode. Defaults to "RGB".
        canvas (CanvasBackend, optional): Canvas backend to use.
            Defaults to PillowCanvasBackend().

    Returns:
        NDArrayUI8: The image with semantic masks drawn into it,
    """
    image = preprocess_image(image, mode=image_mode)
    canvas.create_canvas(image)
    for m, c in zip(
        *preprocess_masks(masks, class_ids, generate_color_map(n_colors))
    ):
        canvas.draw_bitmap(m, c)
    return canvas.as_numpy_image()


def draw_bboxes(
    image: ArrayLike,
    boxes: ArrayLikeFloat,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    canvas: CanvasBackend = PillowCanvasBackend(),
) -> NDArrayUI8:
    """Draws the predicted bounding boxes into the given image.

    Args:
        image (ArrayLike): The image to draw the bboxes into.
        boxes (ArrayLikeFloat): Predicted bounding boxes.
        scores (None | ArrayLikeFloat, optional): Predicted scores.
            Defaults to None.
        class_ids (ArrayLikeInt, optional): Predicted class ids.
            Defaults to None.
        track_ids (ArrayLikeInt, optional): Predicted track ids.
            Defaults to None.
        class_id_mapping (dict[int, str], optional): Mapping from class id to
            name. Defaults to None.
        n_colors (int, optional): Number of colors to use for color palette.
            Defaults to 50.
        image_mode (str, optional): Image Mode. Defaults to "RGB".
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
    image: ArrayLike,
    boxes: ArrayLikeFloat,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
) -> None:
    """Shows the bounding boxes overlayed on the given image.

    Args:
        image (ArrayLike): Background Image
        boxes (ArrayLikeFloat): Boxes to show. Shape [N, 4] with
                                            (x1,y1,x2,y2) as corner convention
        scores (ArrayLikeFloat, optional): Score for each box shape [N]
        class_ids (ArrayLikeInt, optional): Class id for each box shape [N]
        track_ids (ArrayLikeInt, optional): Track id for each box shape [N]
        class_id_mapping (dict[int, str], optional): Mapping to convert
                                                    class id to class name
        n_colors (int, optional): Number of distinct colors used to color the
                                  boxes. Defaults to 50.
        image_mode (str, optional): Image channel mode (RGB or BGR).
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
    """
    image = preprocess_image(image, mode=image_mode)
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
    imshow(img, image_viewer)


def imshow_masks(
    image: ArrayLike,
    masks: ArrayLikeBool,
    class_ids: ArrayLikeInt | None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    canvas: CanvasBackend = PillowCanvasBackend(),
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
) -> None:
    """Shows semantic masks overlayed over the given image.

    Args:
        image (ArrayLike): The image to draw the bboxes into.
        masks (ArrayLikeBool): The semantic masks with the same shape as the
            image.
        class_ids (ArrayLikeInt, optional): Predicted class ids.
            Defaults to None.
        n_colors (int, optional): Number of colors to use for color palette.
            Defaults to 50.
        image_mode (str, optional): Image Mode.. Defaults to "RGB".
        canvas (CanvasBackend, optional): Canvas backend to use.
            Defaults to PillowCanvasBackend().
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
    """
    imshow(
        draw_masks(image, masks, class_ids, n_colors, image_mode, canvas),
        image_viewer,
    )


def imshow_topk_bboxes(
    image: ArrayLike,
    boxes: ArrayLikeFloat,
    scores: ArrayLikeFloat,
    topk: int = 100,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
) -> None:
    """Visualize the 'topk' bounding boxes with highest score.

    Args:
        image (ArrayLike): Background Image
        boxes (ArrayLikeFloat): Boxes to show. Shape [N, 4] with
                                            (x1,y1,x2,y2) as corner convention
        scores (ArrayLikeFloat): Score for each box shape [N]
        topk (int): Number of boxes to visualize
        class_ids (ArrayLikeInt, optional): Class id for each box shape [N]
        track_ids (ArrayLikeInt, optional): Track id for each box shape [N]
        class_id_mapping (dict[int, str], optional): Mapping to convert
                                                    class id to class name
        n_colors (int, optional): Number of distinct colors used to color the
                                  boxes. Defaults to 50.
        image_mode (str, optional): Image channel mode (RGB or BGR).
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().

    """
    scores = array_to_numpy(scores, n_dims=1)
    top_k_idxs = np.argpartition(scores.ravel(), -topk)[-topk:]
    imshow_bboxes(
        image,
        boxes[top_k_idxs],
        scores[top_k_idxs],
        class_ids[top_k_idxs] if class_ids is not None else None,
        track_ids[top_k_idxs] if track_ids is not None else None,
        class_id_mapping,
        n_colors,
        image_mode,
        image_viewer,
    )


def imshow_track_matches(
    key_imgs: list[ArrayLike],
    ref_imgs: list[ArrayLike],
    key_boxes: list[ArrayLikeFloat],
    ref_boxes: list[ArrayLikeFloat],
    key_track_ids: list[ArrayLikeInt],
    ref_track_ids: list[ArrayLikeInt],
    image_mode: str = "RGB",
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
) -> None:
    """Visualize paired bounding boxes successively for batched frame pairs.

    Args:
        key_imgs (list[ArrayLike]): Key Images.
        ref_imgs (list[ArrayLike]): Reference Images.
        key_boxes (list[ArrayLikeFloat]): Predicted Boxes for the key image.
            Shape [N, 4]
        ref_boxes (list[ArrayLikeFloat]): Predicted Boxes for the key image.
            Shape [N, 4]
        key_track_ids (list[ArrayLikeInt]): Predicted ids for the key images.
        ref_track_ids (list[ArrayLikeInt]): Predicted ids for the reference
            images.
        image_mode (str, optional): Color mode if the image. Defaults to "RGB".
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
    """
    key_imgs = arrays_to_numpy(*key_imgs, n_dims=3)
    ref_imgs = arrays_to_numpy(*ref_imgs, n_dims=3)
    key_boxes = arrays_to_numpy(*key_boxes, n_dims=2)
    ref_boxes = arrays_to_numpy(*ref_boxes, n_dims=2)
    key_track_ids = arrays_to_numpy(*key_track_ids, n_dims=1)
    ref_track_ids = arrays_to_numpy(*ref_track_ids, n_dims=1)

    for batch_i, (key_box, ref_box) in enumerate(zip(key_boxes, ref_boxes)):
        target = key_track_ids[batch_i].reshape(-1, 1) == ref_track_ids[
            batch_i
        ].reshape(1, -1)
        for key_i in range(target.shape[0]):
            if target[key_i].sum() == 0:
                continue
            ref_i = np.argmax(target[key_i]).item()
            ref_image = ref_imgs[batch_i]
            key_image = key_imgs[batch_i]

            if ref_image.shape != key_image.shape:
                # Can not stack images together
                imshow_bboxes(
                    key_image,
                    key_box[key_i],
                    image_mode=image_mode,
                    image_viewer=image_viewer,
                )
                imshow_bboxes(
                    ref_image,
                    ref_box[ref_i],
                    image_mode=image_mode,
                    image_viewer=image_viewer,
                )
            else:
                # stack imgs horizontal
                k_img = draw_bboxes(
                    key_image, key_box[batch_i], image_mode=image_mode
                )
                r_img = draw_bboxes(
                    ref_image, ref_box[batch_i], image_mode=image_mode
                )
                stacked_img = np.vstack([k_img, r_img])
                imshow(stacked_img, image_viewer)


# =========================== Pointcloud ===================================
def show_3d(
    scene: Scene3D,
    viewer: PointCloudVisualizerBackend = Open3DVisualizationBackend(
        class_color_mapping=DEFAULT_COLOR_MAPPING
    ),
):
    """Shows a given 3D scene.

    This method shows a 3D visualization of a given 3D scene. Use the viewer
    attribute to use different visualization backends (e.g. open3d)

    Args:
        scene (Scene3D): The 3D scene that should be visualized.
        viewer (PointCloudVisualizerBackend, optional): The Visualization
            backend that should be used to visualize the scene.
            Defaults to Open3DVisualizationBackend.
    """
    viewer.add_scene(scene)
    viewer.show()
    viewer.reset()


def draw_points(
    points_xyz: ArrayLikeFloat,
    colors: ArrayLikeFloat | None = None,
    classes: ArrayLikeInt | None = None,
    instances: ArrayLikeInt | None = None,
    transform: ArrayLikeFloat | None = None,
    scene: Scene3D | None = None,
) -> Scene3D:
    """Adds pointcloud data to a 3D scene for visualization purposes.

    Args:
        points_xyz: xyz coordinates of the points shape [N, 3]
        classes: semantic ids of the points shape [N, 1]
        instances: instance ids of the points shape [N, 1]
        colors: colors of the points shape [N,3] and ranging from  [0,1]
        transform: Optional 4x4 SE3 transform that transforms the point data
            into a static reference frame.
        scene (Scene3D | None): Visualizer that should be used to display the
            data.
    """
    if scene is None:
        scene = Scene3D()

    return scene.add_pointcloud(
        *arrays_to_numpy(points_xyz, colors, n_dims=2),
        *arrays_to_numpy(classes, instances, n_dims=1),
        *arrays_to_numpy(transform, n_dims=2),
    )


def show_points(
    points_xyz: ArrayLikeFloat,
    colors: ArrayLikeFloat | None = None,
    classes: ArrayLikeInt | None = None,
    instances: ArrayLikeInt | None = None,
    transform: ArrayLikeFloat | None = None,
    viewer: PointCloudVisualizerBackend = Open3DVisualizationBackend(
        class_color_mapping=DEFAULT_COLOR_MAPPING
    ),
) -> None:
    """Visualizes a pointcloud with color and semantic information.

    Args:
        points_xyz: xyz coordinates of the points shape [N, 3]
        classes: semantic ids of the points shape [N, 1]
        instances: instance ids of the points shape [N, 1]
        colors: colors of the points shape [N,3] and ranging from  [0,1]
        transform: Optional 4x4 SE3 transform that transforms the point data
            into a static reference frame
        viewer (PointCloudVisualizerBackend, optional): The Visualization
            backend that should be used to visualize the scene.
            Defaults to Open3DVisualizationBackend.
    """
    show_3d(
        draw_points(points_xyz, colors, classes, instances, transform), viewer
    )
