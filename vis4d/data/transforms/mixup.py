"""Mixup data augmentation."""
import random
from typing import TypedDict

import numpy as np
import torch

from vis4d.common.typing import NDArrayF32, NDArrayI32
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_intersection

from .base import Transform
from .resize import get_resize_shape, resize_tensor


class MixupParam(TypedDict):
    """Typed dict for mixup parameters.

    The parameters are used to mixup a pair of items in a batch. Usually, the
    pairs are selected as follows:
        (0, bs - 1), (1, bs - 2), ..., (bs // 2, bs // 2)
    where bs is the batch size. The batch size must be even for mixup to work.
    """

    ratio: float
    im_shape: tuple[int, int]
    im_scale: tuple[float, float]
    other_ori_wh: tuple[int, int]
    other_new_wh: tuple[int, int]
    crop_coord: tuple[int, int, int, int]
    pad_hw: tuple[int, int]
    pad_value: float


@Transform(in_keys=(K.images,), out_keys=("transforms.mixup",))
class GenMixupParameters:
    """Generate the parameters for a mixup operation."""

    def __init__(
        self,
        out_shape: tuple[int, int],
        mixup_ratio_dist: str = "beta",
        alpha: float = 1.0,
        const_ratio: float = 0.5,
        scale_range: tuple[float, float] = (1.0, 1.0),
        clip_inside_image: bool = True,
        pad_value: float = 0.0,
    ) -> None:
        """Init function.

        Args:
            out_shape (tuple[int, int]): Output shape of the mixed up images.
            mixup_ratio_dist (str, optional): Distribution for sampling the
                mixup ratio (i.e., lambda). Options are "beta" and "const".
                Defaults to "beta". If "const", the mixup ratio will be fixed
                to the value of `const_ratio`. Otherwise, the mixup ratio will
                be sampled from a beta distribution with parameters `alpha`.
            alpha (float, optional): Parameter for beta distribution used for
                sampling the mixup ratio (i.e., lambda). Defaults to 1.0.
            const_ratio (float, optional): Constant mixup ratio. Defaults to
                0.5.
            scale_range (tuple[float, float], optional): Range for
                random scale jitter. Defaults to (1.0, 1.0).
            clip_inside_image (bool, optional): Whether to clip the mixed up
                image inside the original image. Defaults to True.
            pad_value (float, optional): Value for padding the mixed up image.
                Defaults to 0.0.
        """
        assert mixup_ratio_dist in {
            "beta",
            "const",
        }, "Mixup ratio distribution must be either 'beta' or 'const'."
        self.mixup_ratio_dist = mixup_ratio_dist
        self.alpha = alpha
        self.const_ratio = const_ratio
        self.out_shape = out_shape
        self.scale_range = scale_range
        self.pad_value = pad_value
        self.clip_inside_image = clip_inside_image

    def __call__(self, images: list[NDArrayF32]) -> list[MixupParam]:
        """Generate parameters for MixUp."""
        batch_size = len(images)
        assert batch_size % 2 == 0, "MixUp only supports even batch size."

        parameter_list = []
        for i in range(batch_size):
            if self.mixup_ratio_dist == "beta":
                ratio = np.random.beta(self.alpha, self.alpha)
            else:
                ratio = self.const_ratio

            h, w = self.out_shape
            ori_img, other_img = images[i], images[batch_size - i - 1]
            ori_h, ori_w = ori_img.shape[1], ori_img.shape[2]
            other_ori_h, other_ori_w = other_img.shape[1], other_img.shape[2]
            other_ori_wh = (other_ori_w, other_ori_h)
            w_i, h_i = get_resize_shape(other_ori_wh, (w, h), keep_ratio=True)
            jit_factor = random.uniform(*self.scale_range)
            h_i, w_i = int(jit_factor * h_i), int(jit_factor * w_i)
            pad_shape = (max(h_i, ori_h), max(w_i, ori_w))

            x_offset, y_offset = 0, 0
            if pad_shape[0] > ori_h:
                y_offset = random.randint(0, pad_shape[0] - ori_h)
            if pad_shape[1] > ori_w:
                x_offset = random.randint(0, pad_shape[1] - ori_w)

            parameters = MixupParam(
                ratio=ratio,
                im_scale=(w_i / other_ori_w, h_i / other_ori_h),
                im_shape=(w_i, h_i),
                other_ori_wh=other_ori_wh,
                other_new_wh=(min(w_i, ori_w), min(h_i, ori_h)),
                pad_hw=pad_shape,
                pad_value=self.pad_value,
                crop_coord=(
                    x_offset,
                    y_offset,
                    x_offset + ori_w,
                    y_offset + ori_h,
                ),
            )
            parameter_list.append(parameters)
        return parameter_list


@Transform(
    in_keys=(K.images, "transforms.mixup"),
    out_keys=(K.images,),
)
class MixupImages:
    """Mixup a batch of images."""

    def __init__(self, interpolation: str = "bilinear") -> None:
        """Init function.

        Args:
            interpolation (str, optional): Interpolation method for resizing
                the other image. Defaults to "bilinear".
        """
        self.interpolation = interpolation

    def __call__(
        self,
        images: list[NDArrayF32],
        mixup_parameters: list[MixupParam],
    ) -> list[NDArrayF32]:
        """Execute image mixup operation."""
        batch_size = len(images)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        mixup_images = []
        for i in range(batch_size):
            j = batch_size - i - 1
            ori_img, other_img = images[i], images[j]
            w_i, h_i = mixup_parameters[i]["im_shape"]
            c = ori_img.shape[-1]

            # resize, scale jitter other image
            other_img_ = torch.from_numpy(other_img).permute(0, 3, 1, 2)
            other_img = (
                resize_tensor(
                    other_img_, (h_i, w_i), interpolation=self.interpolation
                )
                .permute(0, 2, 3, 1)
                .numpy()
            )

            # pad, optionally random crop other image
            padded_img = np.full(
                (1, *mixup_parameters[i]["pad_hw"], c),
                fill_value=mixup_parameters[i]["pad_value"],
                dtype=np.float32,
            )
            padded_img[:, :h_i, :w_i, :] = other_img
            x1_c, y1_c, x2_c, y2_c = mixup_parameters[i]["crop_coord"]
            padded_cropped_img = padded_img[:, y1_c:y2_c, x1_c:x2_c, :]

            # mix ori and other
            ratio = mixup_parameters[i]["ratio"]
            mixup_image = ratio * ori_img + (1 - ratio) * padded_cropped_img
            mixup_images.append(mixup_image)
        return mixup_images


@Transform(
    in_keys=(K.categories, "transforms.mixup"),
    out_keys=(K.categories,),
)
class MixupCategories:
    """Mixup a batch of categories."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.1) -> None:
        """Creates an instance of MixupCategories.

        Args:
            num_classes (int): Number of classes.
            label_smoothing (float, optional): Label smoothing parameter for
                the mixup of categories. Defaults to 0.1.
        """
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def _label_smoothing(
        self,
        cat_1: NDArrayF32,
        cat_2: NDArrayF32,
        ratio: float,
    ) -> NDArrayF32:
        """Apply label smoothing to two category labels."""
        lam = np.array(ratio, dtype=np.float32)
        off_value = np.array(
            self.label_smoothing / self.num_classes, dtype=np.float32
        )
        on_value = np.array(
            1 - self.label_smoothing + off_value, dtype=np.float32
        )
        categories_1: NDArrayF32 = (
            np.zeros((self.num_classes,), dtype=np.float32) + off_value
        )
        categories_2: NDArrayF32 = (
            np.zeros((self.num_classes,), dtype=np.float32) + off_value
        )
        categories_1 = cat_1 * on_value
        categories_2 = cat_2 * on_value
        mixed = categories_1 * lam + categories_2 * (1 - lam)
        return mixed.astype(np.float32)

    def __call__(
        self,
        categories: list[NDArrayF32],
        mixup_parameters: list[MixupParam],
    ) -> list[NDArrayF32]:
        """Execute the categories mixup operation."""
        batch_size = len(categories)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        smooth_categories = [np.empty(0, dtype=np.float32)] * batch_size
        for i in range(batch_size):
            j = batch_size - i - 1
            smooth_categories[i] = self._label_smoothing(
                categories[i], categories[j], mixup_parameters[i]["ratio"]
            )
        return smooth_categories


@Transform(
    in_keys=(K.boxes2d, K.boxes2d_classes, "transforms.mixup"),
    out_keys=(K.boxes2d, K.boxes2d_classes),
)
class MixupBoxes2D:
    """Mixup a batch of boxes."""

    def __init__(self, clip_inside_image: bool = True) -> None:
        """Init function."""
        self.clip_inside_image = clip_inside_image

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        classes_list: list[NDArrayI32],
        mixup_parameters: list[MixupParam],
    ) -> tuple[list[NDArrayF32], list[NDArrayI32]]:
        """Execute the boxes2d mixup operation."""
        batch_size = len(boxes_list)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        mixup_boxes_list = []
        mixup_classes_list = []
        for i in range(batch_size):
            j = batch_size - i - 1
            ori_boxes, other_boxes = boxes_list[i].copy(), boxes_list[j].copy()
            ori_classes, other_classes = (
                classes_list[i].copy(),
                classes_list[j].copy(),
            )
            crop_coord = mixup_parameters[i]["crop_coord"]
            im_scale = mixup_parameters[i]["im_scale"]
            x1_c, y1_c, _, _ = crop_coord

            # adjust boxes to new image size and origin coord
            if len(other_boxes) > 0:
                other_boxes[:, [0, 2]] = (
                    im_scale[0] * other_boxes[:, [0, 2]] - x1_c
                )
                other_boxes[:, [1, 3]] = (
                    im_scale[1] * other_boxes[:, [1, 3]] - y1_c
                )

                # filter boxes outside other image
                crop_box = torch.tensor(crop_coord).unsqueeze(0)
                is_overlap = (
                    bbox_intersection(torch.from_numpy(other_boxes), crop_box)
                    .squeeze(-1)
                    .numpy()
                )
                other_boxes = other_boxes[is_overlap > 0]
                other_classes = other_classes[is_overlap > 0]

                if self.clip_inside_image:
                    new_w, new_h = mixup_parameters[i]["other_new_wh"]
                    other_boxes[:, [0, 2]] = np.clip(
                        other_boxes[:, [0, 2]], 0, new_w
                    )
                    other_boxes[:, [1, 3]] = np.clip(
                        other_boxes[:, [1, 3]], 0, new_h
                    )
                mixup_boxes = np.concatenate((ori_boxes, other_boxes), axis=0)
                mixup_classes = np.concatenate(
                    (ori_classes, other_classes), axis=0
                )
                mixup_boxes_list.append(mixup_boxes)
                mixup_classes_list.append(mixup_classes)
        return mixup_boxes_list, mixup_classes_list


@Transform(
    in_keys=(K.boxes2d, K.boxes2d_track_ids, "transforms.mixup"),
    out_keys=(K.boxes2d_track_ids,),
)
class MixupBoxes2DTrackIds:
    """Mixup a batch of boxes."""

    def __init__(
        self, clip_inside_image: bool = True, max_track_ids: int = 1000
    ) -> None:
        """Init function.

        Args:
            clip_inside_image (bool, optional): Whether to clip the boxes
                inside the image. Defaults to True.
            max_track_ids (int, optional): The maximum number of track ids.
                Defaults to 1000.
        """
        self.clip_inside_image = clip_inside_image
        self.max_track_ids = max_track_ids

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        track_ids_list: list[NDArrayI32],
        mixup_parameters: list[MixupParam],
    ) -> list[NDArrayI32]:
        """Execute the boxes2d mixup operation."""
        batch_size = len(boxes_list)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        mixup_track_ids_list = []
        for i in range(batch_size):
            j = batch_size - i - 1
            other_boxes = boxes_list[j].copy()
            ori_track_ids, other_track_ids = (
                track_ids_list[i].copy(),
                track_ids_list[j].copy(),
            )
            crop_coord = mixup_parameters[i]["crop_coord"]
            im_scale = mixup_parameters[i]["im_scale"]
            x1_c, y1_c, _, _ = crop_coord

            # adjust boxes to new image size and origin coord
            if len(other_boxes) > 0:
                other_boxes[:, [0, 2]] = (
                    im_scale[0] * other_boxes[:, [0, 2]] - x1_c
                )
                other_boxes[:, [1, 3]] = (
                    im_scale[1] * other_boxes[:, [1, 3]] - y1_c
                )
                if (
                    max(other_track_ids) >= self.max_track_ids
                    or max(ori_track_ids) >= self.max_track_ids
                ):
                    raise ValueError(
                        f"Assumes < {self.max_track_ids} labels per image"
                    )
                other_track_ids += self.max_track_ids

                # filter track_ids outside other image
                crop_box = torch.tensor(crop_coord).unsqueeze(0)
                is_overlap = (
                    bbox_intersection(torch.from_numpy(other_boxes), crop_box)
                    .squeeze(-1)
                    .numpy()
                )
                other_track_ids = other_track_ids[is_overlap > 0]
                mixup_track_ids = np.concatenate(
                    (ori_track_ids, other_track_ids), axis=0
                )
                mixup_track_ids_list.append(mixup_track_ids)
        return mixup_track_ids_list
