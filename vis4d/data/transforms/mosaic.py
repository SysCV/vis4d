"""Mosaic transformation.

Modified from mmdetection (https://github.com/open-mmlab/mmdetection).
"""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np

from vis4d.common.typing import NDArrayF32, NDArrayI64
from vis4d.data.const import CommonKeys as K

from .base import Transform
from .crop import _get_keep_mask
from .resize import resize_image


class MosaicParam(TypedDict):
    """Parameters for Mosaic."""

    out_shape: tuple[int, int]
    paste_coords: list[tuple[int, int, int, int]]
    crop_coords: list[tuple[int, int, int, int]]
    im_shapes: list[tuple[int, int]]
    im_scales: list[tuple[float, float]]


def mosaic_combine(
    index: int,
    center: tuple[int, int],
    im_hw: tuple[int, int],
    out_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Compute the mosaic parameters for the image at the current index.

    Index:
    0 = top_left, 1 = top_right, 3 = bottom_left, 4 = bottom_right
    """
    assert index in {0, 1, 2, 3}
    if index == 0:
        # index0 to top left part of image
        x1, y1, x2, y2 = (
            max(center[1] - im_hw[1], 0),
            max(center[0] - im_hw[0], 0),
            center[1],
            center[0],
        )
        crop_coord = (
            im_hw[1] - (x2 - x1),
            im_hw[0] - (y2 - y1),
            im_hw[1],
            im_hw[0],
        )
    elif index == 1:
        # index1 to top right part of image
        x1, y1, x2, y2 = (
            center[1],
            max(center[0] - im_hw[0], 0),
            min(center[1] + im_hw[1], out_shape[1] * 2),
            center[0],
        )
        crop_coord = (
            0,
            im_hw[0] - (y2 - y1),
            min(im_hw[1], x2 - x1),
            im_hw[0],
        )
    elif index == 2:
        # index2 to bottom left part of image
        x1, y1, x2, y2 = (
            max(center[1] - im_hw[1], 0),
            center[0],
            center[1],
            min(out_shape[0] * 2, center[0] + im_hw[0]),
        )
        crop_coord = (
            im_hw[1] - (x2 - x1),
            0,
            im_hw[1],
            min(y2 - y1, im_hw[0]),
        )
    else:
        # index3 to bottom right part of image
        x1, y1, x2, y2 = (
            center[1],
            center[0],
            min(center[1] + im_hw[1], out_shape[1] * 2),
            min(out_shape[0] * 2, center[0] + im_hw[0]),
        )
        crop_coord = 0, 0, min(im_hw[1], x2 - x1), min(y2 - y1, im_hw[0])

    paste_coord = x1, y1, x2, y2
    return paste_coord, crop_coord


@Transform(K.input_hw, ["transforms.mosaic"])
class GenMosaicParameters:
    """Generate the parameters for a mosaic operation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images.
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the dataset.
         3. Sub image will be cropped if image is larger than mosaic patch.

    Args:
        out_shape (tuple[int, int]): The output shape of the mosaic transform.
        center_ratio_range (tuple[float, float]): The range of the ratio of
            the center of the mosaic patch to the output image size.
    """

    NUM_SAMPLES = 4

    def __init__(
        self,
        out_shape: tuple[int, int],
        center_ratio_range: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        """Creates an instance of the class."""
        self.out_shape = out_shape
        self.center_ratio_range = center_ratio_range

    def __call__(self, input_hw: list[tuple[int, int]]) -> list[MosaicParam]:
        """Compute the parameters and put them in the data dict."""
        assert (
            len(input_hw) % self.NUM_SAMPLES == 0
        ), "Input number of images must be a multiple of 4 for Mosaic."
        h, w = self.out_shape
        # mosaic center x, y
        center_y = int(random.uniform(*self.center_ratio_range) * h)
        center_x = int(random.uniform(*self.center_ratio_range) * w)
        center = (center_y, center_x)

        mosaic_params = []
        for i in range(0, len(input_hw), self.NUM_SAMPLES):
            paste_coords, crop_coords, im_scales, im_shapes = [], [], [], []
            for idx, ori_hw in enumerate(input_hw[i : i + self.NUM_SAMPLES]):
                # compute the resize shape
                scale_ratio_i = min(h / ori_hw[0], w / ori_hw[1])
                h_i = int(ori_hw[0] * scale_ratio_i)
                w_i = int(ori_hw[1] * scale_ratio_i)

                # compute the combine parameters
                paste_coord, crop_coord = mosaic_combine(
                    idx, center, (h_i, w_i), self.out_shape
                )
                paste_coords.append(paste_coord)
                crop_coords.append(crop_coord)
                im_shapes.append((h_i, w_i))
                im_scales.append((scale_ratio_i, scale_ratio_i))
            mosaic_params += [
                MosaicParam(
                    out_shape=self.out_shape,
                    paste_coords=paste_coords,
                    crop_coords=crop_coords,
                    im_shapes=im_shapes,
                    im_scales=im_scales,
                )
                for _ in range(self.NUM_SAMPLES)
            ]

        return mosaic_params


@Transform(
    in_keys=[
        K.images,
        "transforms.mosaic.out_shape",
        "transforms.mosaic.paste_coords",
        "transforms.mosaic.crop_coords",
        "transforms.mosaic.im_shapes",
    ],
    out_keys=[K.images, K.input_hw],
)
class MosaicImages:
    """Apply Mosaic to images."""

    NUM_SAMPLES = 4

    def __init__(
        self,
        pad_value: float = 114.0,
        interpolation: str = "bilinear",
        imresize_backend: str = "torch",
    ) -> None:
        """Creates an instance of the class.

        Args:
            pad_value (float): The value to pad the image with. Defaults to
                114.0.
            interpolation (str): Interpolation mode for resizing image.
                Defaults to bilinear.
            imresize_backend (str): One of torch, cv2. Defaults to torch.
        """
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.imresize_backend = imresize_backend
        assert imresize_backend in {
            "torch",
            "cv2",
        }, f"Invalid imresize backend: {imresize_backend}"

    def __call__(
        self,
        images: list[NDArrayF32],
        out_shape: list[tuple[int, int]],
        paste_coords: list[list[tuple[int, int, int, int]]],
        crop_coords: list[list[tuple[int, int, int, int]]],
        im_shapes: list[list[tuple[int, int]]],
    ) -> tuple[list[NDArrayF32], list[tuple[int, int]]]:
        """Resize an image of dimensions [N, H, W, C]."""
        h, w = out_shape[0]
        c = images[0].shape[-1]

        mosaic_imgs = []
        for i in range(0, len(images), self.NUM_SAMPLES):
            mosaic_img = np.full(
                (1, h * 2, w * 2, c), self.pad_value, dtype=np.float32
            )
            for idx, img in enumerate(images[i : i + self.NUM_SAMPLES]):
                # resize current image
                h_i, w_i = im_shapes[i][idx]
                img_ = resize_image(
                    img,
                    (h_i, w_i),
                    self.interpolation,
                    backend=self.imresize_backend,
                )

                x1_p, y1_p, x2_p, y2_p = paste_coords[i][idx]
                x1_c, y1_c, x2_c, y2_c = crop_coords[i][idx]

                # crop and paste image
                mosaic_img[:, y1_p:y2_p, x1_p:x2_p, :] = img_[
                    :, y1_c:y2_c, x1_c:x2_c, :
                ]
            mosaic_imgs += [mosaic_img for _ in range(self.NUM_SAMPLES)]
        return mosaic_imgs, [(m.shape[1], m.shape[2]) for m in mosaic_imgs]


@Transform(
    in_keys=[
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        "transforms.mosaic.paste_coords",
        "transforms.mosaic.crop_coords",
        "transforms.mosaic.im_scales",
    ],
    out_keys=[K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids],
)
class MosaicBoxes2D:
    """Apply Mosaic to a list of 2D bounding boxes."""

    NUM_SAMPLES = 4

    def __init__(
        self, clip_inside_image: bool = True, max_track_ids: int = 1000
    ) -> None:
        """Creates an instance of the class.

        Args:
            clip_inside_image (bool): Whether to clip the boxes to be inside
                the image. Defaults to True.
            max_track_ids (int): The maximum number of track ids. Defaults to
                1000.
        """
        self.clip_inside_image = clip_inside_image
        self.max_track_ids = max_track_ids

    def __call__(
        self,
        boxes: list[NDArrayF32],
        classes: list[NDArrayI64],
        track_ids: list[NDArrayI64] | None,
        paste_coords: list[list[tuple[int, int, int, int]]],
        crop_coords: list[list[tuple[int, int, int, int]]],
        im_scales: list[list[tuple[float, float]]],
    ) -> tuple[list[NDArrayF32], list[NDArrayI64], list[NDArrayI64] | None]:
        """Apply Mosaic to 2D bounding boxes."""
        new_boxes, new_classes = [], []
        new_track_ids: list[NDArrayI64] | None = (
            [] if track_ids is not None else None
        )
        for i in range(0, len(boxes), self.NUM_SAMPLES):
            for idx in range(self.NUM_SAMPLES):
                j = i + idx

                x1_p, y1_p, x2_p, y2_p = paste_coords[i][idx]
                x1_c, y1_c, _, _ = crop_coords[i][idx]

                pw = x1_p - x1_c
                ph = y1_p - y1_c
                boxes[j][:, [0, 2]] = (
                    im_scales[i][idx][1] * boxes[j][:, [0, 2]] + pw
                )
                boxes[j][:, [1, 3]] = (
                    im_scales[i][idx][0] * boxes[j][:, [1, 3]] + ph
                )

                keep_mask = _get_keep_mask(
                    boxes[j], np.array([x1_p, y1_p, x2_p, y2_p])
                )
                boxes[j] = boxes[j][keep_mask]
                classes[j] = classes[j][keep_mask]
                if track_ids is not None:
                    track_ids[j] = track_ids[j][keep_mask].copy()
                    if len(track_ids[j]) > 0:
                        if max(track_ids[j]) >= self.max_track_ids:
                            raise ValueError(
                                f"Track id exceeds maximum track id"
                                f"{self.max_track_ids}!"
                            )
                        track_ids[j] += self.max_track_ids * idx

                if self.clip_inside_image:
                    boxes[j][:, [0, 2]] = boxes[j][:, [0, 2]].clip(x1_p, x2_p)
                    boxes[j][:, [1, 3]] = boxes[j][:, [1, 3]].clip(y1_p, y2_p)
            new_boxes += [
                np.concatenate(boxes[i : i + self.NUM_SAMPLES])
                for _ in range(self.NUM_SAMPLES)
            ]
            new_classes += [
                np.concatenate(classes[i : i + self.NUM_SAMPLES])
                for _ in range(self.NUM_SAMPLES)
            ]
            if track_ids is not None:
                assert new_track_ids is not None
                new_track_ids += [
                    np.concatenate(track_ids[i : i + self.NUM_SAMPLES])
                    for _ in range(self.NUM_SAMPLES)
                ]
        return new_boxes, new_classes, new_track_ids
