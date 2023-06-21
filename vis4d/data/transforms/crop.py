"""Crop transformation."""
from __future__ import annotations

from collections.abc import Callable
from typing import List, Tuple, TypedDict, Union

import numpy as np
import torch

from vis4d.common.logging import rank_zero_warn
from vis4d.common.typing import NDArrayBool, NDArrayF32, NDArrayI32, NDArrayUI8
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_intersection

from .base import Transform

CropShape = Union[
    Tuple[float, float],
    Tuple[int, int],
    List[Tuple[float, float]],
    List[Tuple[int, int]],
]
CropFunc = Callable[[int, int, CropShape], Tuple[int, int]]


class CropParam(TypedDict):
    """Parameters for Crop."""

    crop_box: NDArrayI32
    keep_mask: NDArrayBool


def absolute_crop(im_h: int, im_w: int, shape: CropShape) -> tuple[int, int]:
    """Absolute crop."""
    assert isinstance(shape, tuple)
    assert shape[0] > 0 and shape[1] > 0
    return (min(int(shape[0]), im_h), min(int(shape[1]), im_w))


def absolute_range_crop(
    im_h: int, im_w: int, shape: CropShape
) -> tuple[int, int]:
    """Absolute range crop."""
    assert isinstance(shape, list)
    assert len(shape) == 2
    assert shape[1][0] >= shape[0][0]
    assert shape[1][1] >= shape[0][1]

    for crop in shape:
        assert crop[0] > 0 and crop[1] > 0
        shape_min: tuple[int, int] = (int(shape[0][0]), int(shape[0][1]))
        shape_max: tuple[int, int] = (int(shape[1][0]), int(shape[1][1]))

    crop_h = np.random.randint(
        min(im_h, shape_min[0]), min(im_h, shape_max[0]) + 1
    )
    crop_w = np.random.randint(
        min(im_w, shape_min[1]), min(im_w, shape_max[1]) + 1
    )
    return int(crop_h), int(crop_w)


def relative_crop(im_h: int, im_w: int, shape: CropShape) -> tuple[int, int]:
    """Relative crop."""
    assert isinstance(shape, tuple)
    assert 0 < shape[0] <= 1 and 0 < shape[1] <= 1
    crop_h, crop_w = shape
    return int(im_h * crop_h + 0.5), int(im_w * crop_w + 0.5)


def relative_range_crop(
    im_h: int, im_w: int, shape: CropShape
) -> tuple[int, int]:
    """Relative range crop."""
    assert isinstance(shape, list)
    assert len(shape) == 2
    assert shape[1][0] >= shape[0][0]
    assert shape[1][1] >= shape[0][1]
    for crop in shape:
        assert 0 < crop[0] <= 1 and 0 < crop[1] <= 1
    scale_min: tuple[float, float] = shape[0]
    scale_max: tuple[float, float] = shape[1]

    crop_h = np.random.rand() * (scale_max[0] - scale_min[0]) + scale_min[0]
    crop_w = np.random.rand() * (scale_max[1] - scale_min[1]) + scale_min[1]
    return int(im_h * crop_h + 0.5), int(im_w * crop_w + 0.5)


@Transform(
    in_keys=[K.input_hw, K.boxes2d, K.seg_masks],
    out_keys="transforms.crop",
)
class GenCropParameters:
    """Generate the parameters for a crop operation."""

    def __init__(
        self,
        shape: CropShape,
        crop_func: CropFunc = absolute_crop,
        allow_empty_crops: bool = True,
        recompute_boxes2d: bool = False,
        cat_max_ratio: float = 1.0,
    ) -> None:
        """Creates an instance of the class.

        Args:
            shape (CropShape): Image shape to be cropped to.
            crop_func (CropFunc, optional): Function used to generate the size
                of the crop. Defaults to absolute_crop.
            allow_empty_crops (bool, optional): Allow crops which result in
                empty labels. Defaults to True.
            recompute_boxes2d (bool, optional): Recompute the bounding boxes
                after cropping instance masks. Defaults to False.
            cat_max_ratio (float, optional): Maximum ratio of a particular
                class in segmentation masks after cropping. Defaults to 1.0.
        """
        self.shape = shape
        self.crop_func = crop_func
        self.cat_max_ratio = cat_max_ratio
        self.allow_empty_crops = allow_empty_crops
        self.recompute_boxes2d = recompute_boxes2d

    def _get_crop(
        self, im_h: int, im_w: int, boxes: NDArrayF32 | None = None
    ) -> tuple[NDArrayI32, NDArrayBool]:
        """Get the crop parameters."""
        crop_size = self.crop_func(im_h, im_w, self.shape)
        crop_box = _sample_crop(im_h, im_w, crop_size)
        keep_mask = (
            _get_keep_mask(boxes, crop_box)
            if boxes is not None
            else np.array([])
        )
        return crop_box, keep_mask

    def __call__(
        self,
        input_hw_list: list[tuple[int, int]],
        boxes_list: list[NDArrayF32 | None],
        masks_list: list[NDArrayUI8 | None],
    ) -> list[CropParam]:
        """Compute the parameters and put them in the data dict."""
        im_h, im_w = input_hw_list[0]
        boxes = boxes_list[0]
        masks = masks_list[0]

        crop_box, keep_mask = self._get_crop(im_h, im_w, boxes)
        if (boxes is not None and len(boxes) > 0) or self.cat_max_ratio != 1.0:
            # resample crop if conditions not satisfied
            found_crop = False
            for _ in range(10):
                # try resampling 10 times, otherwise use last crop
                if (self.allow_empty_crops or keep_mask.sum() != 0) and (
                    masks is None
                    or _check_seg_max_cat(masks, crop_box, self.cat_max_ratio)
                ):
                    found_crop = True
                    break
                crop_box, keep_mask = self._get_crop(im_h, im_w, boxes)
            if not found_crop:
                rank_zero_warn("Random crop not found within 10 resamples.")

        crop_params = [
            CropParam(crop_box=crop_box, keep_mask=keep_mask)
        ] * len(input_hw_list)

        return crop_params


@Transform([K.images, "transforms.crop.crop_box"], [K.images, K.input_hw])
class CropImages:
    """Crop Images."""

    def __call__(
        self, images: list[NDArrayF32], crop_box_list: list[NDArrayI32]
    ) -> tuple[list[NDArrayF32], list[tuple[int, int]]]:
        """Crop a list of image of dimensions [N, H, W, C].

        Args:
            images (list[NDArrayF32]): The list of image.
            crop_box (list[NDArrayI32]): The list of box to crop.

        Returns:
            list[NDArrayF32]: List of cropped image according to parameters.
        """
        input_hw_list = []
        for i, (image, crop_box) in enumerate(zip(images, crop_box_list)):
            h, w = image.shape[1], image.shape[2]
            x1, y1, x2, y2 = crop_box
            crop_w, crop_h = x2 - x1, y2 - y1
            image = image[:, y1:y2, x1:x2, :]
            input_hw = (min(crop_h, h), min(crop_w, w))

            images[i] = image
            input_hw_list.append(input_hw)
        return images, input_hw_list


@Transform(
    in_keys=[
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        "transforms.crop.crop_box",
        "transforms.crop.keep_mask",
    ],
    out_keys=[K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids],
)
class CropBoxes2D:
    """Crop 2D bounding boxes."""

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        classes_list: list[NDArrayI32],
        track_ids_list: list[NDArrayI32 | None],
        crop_box_list: list[NDArrayI32],
        keep_mask_list: list[NDArrayBool],
    ) -> tuple[list[NDArrayF32], list[NDArrayI32], list[NDArrayI32 | None]]:
        """Crop 2D bounding boxes.

        Args:
            boxes_list (list[NDArrayF32]): The list of bounding boxes to be
                cropped.
            classes_list (list[NDArrayI32]): The list of the corresponding
                classes.
            crop_box_list (list[NDArrayI32]): The list of box to crop.
            keep_mask (list[NDArrayBool]): Which boxes to keep.
            track_ids (list[NDArrayI32] | None, optional): The list of
                corresponding tracking IDs. Defaults to None.

        Returns:
            tuple[list[NDArrayF32], list[NDArrayI32], list[NDArrayI32] | None]:
                List of cropped bounding boxes according to parameters.
        """
        for i, (boxes, classes, track_ids, crop_box, keep_mask) in enumerate(
            zip(
                boxes_list,
                classes_list,
                track_ids_list,
                crop_box_list,
                keep_mask_list,
            )
        ):
            x1, y1 = crop_box[:2]
            boxes -= np.array([x1, y1, x1, y1])

            boxes_list[i] = boxes[keep_mask]
            classes_list[i] = classes[keep_mask]

            if track_ids is not None:
                track_ids_list[i] = track_ids[keep_mask]

        return boxes_list, classes_list, track_ids_list


@Transform([K.seg_masks, "transforms.crop.crop_box"], K.seg_masks)
class CropSegMasks:
    """Crop segmentation masks."""

    def __call__(
        self, masks_list: list[NDArrayUI8], crop_box_list: list[NDArrayI32]
    ) -> list[NDArrayUI8]:
        """Crop masks."""
        for i, (masks, crop_box) in enumerate(zip(masks_list, crop_box_list)):
            x1, y1, x2, y2 = crop_box
            masks_list[i] = masks[y1:y2, x1:x2]
        return masks_list


@Transform([K.depth_maps, "transforms.crop.crop_box"], K.depth_maps)
class CropDepthMaps:
    """Crop depth maps."""

    def __call__(
        self, depth_maps: list[NDArrayF32], crop_box_list: list[NDArrayI32]
    ) -> list[NDArrayF32]:
        """Crop depth maps."""
        for i, (depth_map, crop_box) in enumerate(
            zip(depth_maps, crop_box_list)
        ):
            x1, y1, x2, y2 = crop_box
            depth_maps[i] = depth_map[y1:y2, x1:x2]
        return depth_maps


@Transform([K.optical_flows, "transforms.crop.crop_box"], K.optical_flows)
class CropOpticalFlows:
    """Crop optical flows."""

    def __call__(
        self, optical_flows: list[NDArrayF32], crop_box_list: NDArrayI32
    ) -> list[NDArrayF32]:
        """Crop optical flows."""
        for i, (optical_flow, crop_box) in enumerate(
            zip(optical_flows, crop_box_list)
        ):
            x1, y1, x2, y2 = crop_box
            optical_flows[i] = optical_flow[y1:y2, x1:x2]
        return optical_flows


@Transform([K.intrinsics, "transforms.crop.crop_box"], K.intrinsics)
class CropIntrinsics:
    """Crop Intrinsics."""

    def __call__(
        self,
        intrinsics_list: list[NDArrayF32],
        crop_box_list: list[NDArrayI32],
    ) -> list[NDArrayF32]:
        """Crop camera intrinsics."""
        for i, crop_box in enumerate(crop_box_list):
            x1, y1 = crop_box[:2]
            intrinsics_list[i][0, 2] -= x1
            intrinsics_list[i][1, 2] -= y1
        return intrinsics_list


def _sample_crop(
    im_h: int, im_w: int, crop_size: tuple[int, int]
) -> NDArrayI32:
    """Sample crop parameters according to config."""
    margin_h = max(im_h - crop_size[0], 0)
    margin_w = max(im_w - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    return np.array([crop_x1, crop_y1, crop_x2, crop_y2])


def _get_keep_mask(boxes: NDArrayF32, crop_box: NDArrayI32) -> NDArrayBool:
    """Get mask for 2D annotations to keep."""
    if len(boxes) == 0:
        return np.array([])
    # will be better to compute mask intersection (if exists) instead
    overlap = bbox_intersection(
        torch.tensor(boxes), torch.tensor(crop_box).unsqueeze(0)
    ).numpy()
    return overlap.squeeze(-1) > 0


def _check_seg_max_cat(
    masks: NDArrayUI8, crop_box: NDArrayI32, cat_max_ratio: float
) -> bool:
    """Check if any category occupies more than cat_max_ratio.

    Args:
        masks (NDArrayUI8): Segmentation masks.
        crop_box (NDArrayI32): The box to crop.
        cat_max_ratio (float): Maximum category ratio.

    Returns:
        bool: True if no category occupies more than cat_max_ratio.
    """
    x1, y1, x2, y2 = crop_box
    crop_masks = masks[y1:y2, x1:x2]
    cls_ids, cnts = np.unique(crop_masks, return_counts=True)
    cnts = cnts[cls_ids != 255]
    keep_mask = len(cnts) > 1 and cnts.max() / cnts.sum() < cat_max_ratio
    return keep_mask
