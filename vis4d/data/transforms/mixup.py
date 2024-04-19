"""Mixup data augmentation."""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch

from vis4d.common.typing import NDArrayF32, NDArrayI64
from vis4d.data.const import CommonKeys as K
from vis4d.op.box.box2d import bbox_intersection

from .base import Transform
from .resize import get_resize_shape, resize_image


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
    other_ori_hw: tuple[int, int]
    other_new_hw: tuple[int, int]
    crop_coord: tuple[int, int, int, int]
    pad_hw: tuple[int, int]
    pad_value: float


@Transform(in_keys=(K.images,), out_keys=("transforms.mixup",))
class GenMixupParameters:
    """Generate the parameters for a mixup operation."""

    NUM_SAMPLES = 2

    def __init__(
        self,
        out_shape: tuple[int, int],
        mixup_ratio_dist: str = "beta",
        alpha: float = 1.0,
        const_ratio: float = 0.5,
        scale_range: tuple[float, float] = (1.0, 1.0),
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
            pad_value (float, optional): Value for padding the mixed up image.
                Defaults to 0.0.
        """
        assert mixup_ratio_dist in {
            "beta",
            "const",
        }, "Mixup ratio distribution must be either 'beta' or 'const'."
        self.out_shape = out_shape
        self.mixup_ratio_dist = mixup_ratio_dist
        self.alpha = alpha
        self.const_ratio = const_ratio
        self.scale_range = scale_range
        self.pad_value = pad_value

    def __call__(self, images: list[NDArrayF32]) -> list[MixupParam]:
        """Generate parameters for MixUp."""
        batch_size = len(images)
        assert batch_size % 2 == 0, "MixUp only supports even batch size."

        if self.mixup_ratio_dist == "beta":
            ratio = np.random.beta(self.alpha, self.alpha)
        else:
            ratio = self.const_ratio
        jit_factor = random.uniform(*self.scale_range)

        h, w = self.out_shape
        ori_img, other_img = images[0], images[1]
        ori_h, ori_w = ori_img.shape[1], ori_img.shape[2]
        other_ori_h, other_ori_w = other_img.shape[1], other_img.shape[2]
        other_ori_hw = (other_ori_h, other_ori_w)
        h_i, w_i = get_resize_shape(other_ori_hw, (h, w), keep_ratio=True)
        h_i, w_i = int(jit_factor * h_i), int(jit_factor * w_i)
        pad_shape = (max(h_i, ori_h), max(w_i, ori_w))

        x_offset, y_offset = 0, 0
        if pad_shape[0] > ori_h:
            y_offset = random.randint(0, pad_shape[0] - ori_h)
        if pad_shape[1] > ori_w:
            x_offset = random.randint(0, pad_shape[1] - ori_w)

        parameter_list = [
            MixupParam(
                ratio=ratio,
                im_scale=(h_i / other_ori_h, w_i / other_ori_w),
                im_shape=(h_i, w_i),
                other_ori_hw=other_ori_hw,
                other_new_hw=(min(h_i, ori_h), min(w_i, ori_w)),
                pad_hw=pad_shape,
                pad_value=self.pad_value,
                crop_coord=(
                    x_offset,
                    y_offset,
                    x_offset + ori_w,
                    y_offset + ori_h,
                ),
            )
            for _ in range(batch_size)
        ]
        return parameter_list


@Transform(in_keys=(K.images, "transforms.mixup"), out_keys=(K.images,))
class MixupImages:
    """Mixup a batch of images."""

    NUM_SAMPLES = 2

    def __init__(
        self, interpolation: str = "bilinear", imresize_backend: str = "torch"
    ) -> None:
        """Init function.

        Args:
            interpolation (str, optional): Interpolation method for resizing
                the other image. Defaults to "bilinear".
            imresize_backend (str): One of torch, cv2. Defaults to torch.
        """
        self.interpolation = interpolation
        self.imresize_backend = imresize_backend
        assert imresize_backend in {
            "torch",
            "cv2",
        }, f"Invalid imresize backend: {imresize_backend}"

    def __call__(
        self, images: list[NDArrayF32], mixup_parameters: list[MixupParam]
    ) -> list[NDArrayF32]:
        """Execute image mixup operation."""
        batch_size = len(images)
        assert (
            batch_size % self.NUM_SAMPLES == 0
        ), "Batch size must be even for mixup!"

        mixup_images = []
        for i in range(0, batch_size, self.NUM_SAMPLES):
            j = i + 1
            ori_img, other_img = images[i], images[j]
            h_i, w_i = mixup_parameters[i]["im_shape"]
            c = ori_img.shape[-1]

            # resize, scale jitter other image
            other_img = resize_image(
                other_img,
                (h_i, w_i),
                self.interpolation,
                backend=self.imresize_backend,
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
            mixup_images += [mixup_image for _ in range(self.NUM_SAMPLES)]
        return mixup_images


@Transform(
    in_keys=(K.categories, "transforms.mixup"), out_keys=(K.categories,)
)
class MixupCategories:
    """Mixup a batch of categories."""

    NUM_SAMPLES = 2

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
        assert (
            batch_size % self.NUM_SAMPLES == 0
        ), "Batch size must be even for mixup!"

        smooth_categories = [np.empty(0, dtype=np.float32)] * batch_size
        for i in range(0, batch_size, self.NUM_SAMPLES):
            j = i + 1
            smooth_categories[i] = self._label_smoothing(
                categories[i], categories[j], mixup_parameters[i]["ratio"]
            )
            smooth_categories[j] = smooth_categories[i]
        return smooth_categories


@Transform(
    in_keys=(
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_track_ids,
        "transforms.mixup",
    ),
    out_keys=(K.boxes2d, K.boxes2d_classes, K.boxes2d_track_ids),
)
class MixupBoxes2D:
    """Mixup a batch of boxes."""

    NUM_SAMPLES = 2

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
        boxes_list: list[NDArrayF32],
        classes_list: list[NDArrayI64],
        track_ids_list: list[NDArrayI64] | None,
        mixup_parameters: list[MixupParam],
    ) -> tuple[list[NDArrayF32], list[NDArrayI64], list[NDArrayI64] | None]:
        """Execute the boxes2d mixup operation."""
        batch_size = len(boxes_list)
        assert (
            batch_size % self.NUM_SAMPLES == 0
        ), "Batch size must be even for mixup!"

        mixup_boxes_list = []
        mixup_classes_list = []
        mixup_track_ids_list: list[NDArrayI64] | None = (
            [] if track_ids_list is not None else None
        )
        for i in range(0, batch_size, self.NUM_SAMPLES):
            j = i + 1
            ori_boxes, other_boxes = boxes_list[i].copy(), boxes_list[j].copy()
            ori_classes, other_classes = (
                classes_list[i].copy(),
                classes_list[j].copy(),
            )

            crop_coord = mixup_parameters[i]["crop_coord"]
            im_scale = mixup_parameters[i]["im_scale"]
            x1_c, y1_c, _, _ = crop_coord

            if len(other_boxes) == 0:
                continue
            # adjust boxes to new image size and origin coord
            other_boxes[:, [0, 2]] = (
                im_scale[1] * other_boxes[:, [0, 2]] - x1_c
            )
            other_boxes[:, [1, 3]] = (
                im_scale[0] * other_boxes[:, [1, 3]] - y1_c
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

            # mixup track ids if available
            if track_ids_list is not None:
                assert mixup_track_ids_list is not None
                ori_track_ids = track_ids_list[i].copy()
                other_track_ids = track_ids_list[j].copy()
                if (
                    len(ori_track_ids) > 0
                    and max(ori_track_ids) >= self.max_track_ids
                ) or (
                    len(other_track_ids) > 0
                    and max(other_track_ids) >= self.max_track_ids
                ):
                    raise ValueError(
                        f"Track id exceeds maximum track id"
                        f"{self.max_track_ids}!"
                    )
                other_track_ids += self.max_track_ids
                other_track_ids = other_track_ids[is_overlap > 0]
                mixup_track_ids: NDArrayI64 = np.concatenate(
                    (ori_track_ids, other_track_ids), 0
                )
                mixup_track_ids_list += [
                    mixup_track_ids for _ in range(self.NUM_SAMPLES)
                ]

            if self.clip_inside_image:
                new_h, new_w = mixup_parameters[i]["other_new_hw"]
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
            mixup_boxes_list += [mixup_boxes for _ in range(self.NUM_SAMPLES)]
            mixup_classes_list += [
                mixup_classes for _ in range(self.NUM_SAMPLES)
            ]
        return mixup_boxes_list, mixup_classes_list, mixup_track_ids_list
