"""Function interface for image visualization functions."""

from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import (
    ArrayLike,
    ArrayLikeBool,
    ArrayLikeFloat,
    ArrayLikeInt,
    NDArrayF32,
    NDArrayUI8,
)

from ..util import generate_color_map
from .canvas import CanvasBackend, PillowCanvasBackend
from .util import (
    preprocess_boxes,
    preprocess_boxes3d,
    preprocess_image,
    preprocess_masks,
    project_point,
)
from .viewer import ImageViewerBackend, MatplotlibImageViewer


def imshow(
    image: ArrayLike,
    image_mode: str = "RGB",
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
    file_path: str | None = None,
) -> None:
    """Shows a single image.

    Args:
        image (NDArrayNumber): The image to show.
        image_mode (str, optional): Image Mode. Defaults to "RGB".
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
        file_path (str): The path to save the image to. Defaults to None.
    """
    image = preprocess_image(image, image_mode)
    image_viewer.show_images([image])

    if file_path is not None:
        image_viewer.save_images([image], [file_path])


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
    box_width: int = 1,
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
        box_width (int, optional): Width of the box border. Defaults to 1.
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
        canvas.draw_box(corners, color, box_width)
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
    box_width: int = 1,
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
    file_path: str | None = None,
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
        box_width (int, optional): Width of the box border. Defaults to 1.
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
        file_path (str): The path to save the image to. Defaults to None.
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
        box_width,
    )
    imshow(img, image_mode, image_viewer, file_path)


def draw_bbox3d(
    image: NDArrayUI8,
    boxes3d: ArrayLikeFloat,
    intrinsics: NDArrayF32,
    extrinsics: NDArrayF32 | None = None,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    canvas: CanvasBackend = PillowCanvasBackend(),
    width: int = 4,
    camera_near_clip: float = 0.15,
) -> NDArrayUI8:
    """Draw 3D box onto image."""
    image = preprocess_image(image, image_mode)
    image_hw = (image.shape[0], image.shape[1])
    boxes3d_data = preprocess_boxes3d(
        image_hw,
        boxes3d,
        intrinsics,
        extrinsics,
        scores,
        class_ids,
        track_ids,
        color_palette=generate_color_map(n_colors),
        class_id_mapping=class_id_mapping,
    )
    canvas.create_canvas(image)

    for _, corners, label, color, _ in zip(*boxes3d_data):
        canvas.draw_box_3d(corners, color, intrinsics, width, camera_near_clip)

        selected_corner = project_point(corners[0], intrinsics)
        canvas.draw_text((selected_corner[0], selected_corner[1]), label)

    return canvas.as_numpy_image()


def imshow_bboxes3d(
    image: ArrayLike,
    boxes3d: ArrayLikeFloat,
    intrinsics: NDArrayF32,
    extrinsics: NDArrayF32 | None = None,
    scores: None | ArrayLikeFloat = None,
    class_ids: None | ArrayLikeInt = None,
    track_ids: None | ArrayLikeInt = None,
    class_id_mapping: None | dict[int, str] = None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
    file_path: str | None = None,
) -> None:
    """Show image with bounding boxes."""
    image = preprocess_image(image, mode=image_mode)
    img = draw_bbox3d(
        image,
        boxes3d,
        intrinsics,
        extrinsics,
        scores,
        class_ids,
        track_ids,
        class_id_mapping=class_id_mapping,
        n_colors=n_colors,
        image_mode=image_mode,
    )
    imshow(img, image_mode, image_viewer, file_path)


def imshow_masks(
    image: ArrayLike,
    masks: ArrayLikeBool,
    class_ids: ArrayLikeInt | None,
    n_colors: int = 50,
    image_mode: str = "RGB",
    canvas: CanvasBackend = PillowCanvasBackend(),
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
    file_path: str | None = None,
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
        file_path (str): The path to save the image to. Defaults to None.
    """
    imshow(
        draw_masks(image, masks, class_ids, n_colors, image_mode, canvas),
        image_mode,
        image_viewer,
        file_path,
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
    box_width: int = 1,
    image_viewer: ImageViewerBackend = MatplotlibImageViewer(),
    file_path: str | None = None,
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
        box_width (int, optional): Width of the box border. Defaults to 1.
        image_viewer (ImageViewerBackend, optional): The Image viewer backend
            to use. Defaults to MatplotlibImageViewer().
        file_path (str): The path to save the image to. Defaults to None.

    """
    scores = array_to_numpy(scores, n_dims=1, dtype=np.float32)
    top_k_idxs = np.argpartition(scores.ravel(), -topk)[-topk:]

    boxes_np = array_to_numpy(boxes, n_dims=2, dtype=np.float32)
    class_ids_np = array_to_numpy(class_ids, n_dims=1, dtype=np.int32)
    track_ids_np = array_to_numpy(track_ids, n_dims=1, dtype=np.int32)
    imshow_bboxes(
        image,
        boxes_np[top_k_idxs],
        scores[top_k_idxs],
        class_ids_np[top_k_idxs] if class_ids_np is not None else None,
        track_ids_np[top_k_idxs] if track_ids_np is not None else None,
        class_id_mapping,
        n_colors,
        image_mode,
        box_width,
        image_viewer,
        file_path,
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
    file_path: str | None = None,
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
        file_path (str): The path to save the image to. Defaults to None.
    """
    key_imgs_np = tuple(
        array_to_numpy(img, n_dims=3, dtype=np.float32) for img in key_imgs
    )
    ref_imgs_np = tuple(
        array_to_numpy(img, n_dims=3, dtype=np.float32) for img in ref_imgs
    )
    key_boxes_np = tuple(
        array_to_numpy(b, n_dims=2, dtype=np.float32) for b in key_boxes
    )
    ref_boxes_np = tuple(
        array_to_numpy(b, n_dims=2, dtype=np.float32) for b in ref_boxes
    )
    key_track_ids_np = tuple(
        array_to_numpy(t, n_dims=1, dtype=np.int32) for t in key_track_ids
    )
    ref_track_ids_np = tuple(
        array_to_numpy(t, n_dims=1, dtype=np.int32) for t in ref_track_ids
    )

    for batch_i, (key_box, ref_box) in enumerate(
        zip(key_boxes_np, ref_boxes_np)
    ):
        target = key_track_ids_np[batch_i].reshape(-1, 1) == ref_track_ids_np[
            batch_i
        ].reshape(1, -1)
        for key_i in range(target.shape[0]):
            if target[key_i].sum() == 0:
                continue
            ref_i = np.argmax(target[key_i]).item()
            ref_image = ref_imgs_np[batch_i]
            key_image = key_imgs_np[batch_i]

            if ref_image.shape != key_image.shape:
                # Can not stack images together
                imshow_bboxes(
                    key_image,
                    key_box[key_i],
                    image_mode=image_mode,
                    image_viewer=image_viewer,
                    file_path=file_path,
                )
                imshow_bboxes(
                    ref_image,
                    ref_box[ref_i],
                    image_mode=image_mode,
                    image_viewer=image_viewer,
                    file_path=file_path,
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
                imshow(stacked_img, image_mode, image_viewer, file_path)
