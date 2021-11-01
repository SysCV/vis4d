"""VisT augmentations."""
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from vist.common.bbox.utils import bbox_intersection
from vist.data.utils import transform_bbox
from vist.struct import (
    Boxes2D,
    Boxes3D,
    Images,
    InputSample,
    Intrinsics,
    Masks,
)

from .base import AugParams, BaseAugmentation, BaseAugmentationConfig


class ResizeConfig(BaseAugmentationConfig):
    """Resize augmentation config.

    shape: Image shape to be resized to in (H, W) format. In
    multiscale mode 'list', shape represents the list of possible
    shapes for resizing.
    interpolation: Interpolation method. One of ["nearest", "bilinear",
    "bicubic"]
    keep_ratio: If aspect ratio of original image should be kept, the
    new height will be scaled according to the new width and the
    aspect ratio of the original image as:
    new_h = new_w / (orginal_w / original_h)
    multiscale_mode: one of [range, list],
    scale_range: Range of sampled image scales in range mode, e.g.
    (0.8, 1.2), indicating minimum of 0.8 * shape and maximum of
    1.2 * shape.
    return_transform: If the transform should be returned in matrix
    format.
    """

    shape: Union[Tuple[int, int], List[Tuple[int, int]]]
    keep_ratio: bool = False
    multiscale_mode: str = "range"
    scale_range: Tuple[float, float] = (1.0, 1.0)
    interpolation: str = "bilinear"


class Resize(BaseAugmentation):
    """Resize augmentation class."""

    def __init__(
        self,
        cfg: BaseAugmentationConfig,
    ) -> None:
        """Init function.

        Args:
            cfg: Augmentation config.
        """
        super().__init__(cfg)
        self.cfg: ResizeConfig = ResizeConfig(**cfg.dict())
        self.shape = self.cfg.shape
        self.keep_ratio = self.cfg.keep_ratio
        self.multiscale_mode = self.cfg.multiscale_mode
        assert self.multiscale_mode in ["list", "range"]
        self.scale_range = self.cfg.scale_range
        if self.multiscale_mode == "list":
            assert isinstance(
                self.shape, list
            ), "Specify shape as list when using multiscale mode list."
            assert len(self.shape) >= 1
            # pylint: disable=unsubscriptable-object
            self.shape = [(int(s[0]), int(s[1])) for s in self.shape]
        else:
            assert isinstance(
                self.shape, tuple
            ), "Specify shape as tuple when using multiscale mode range."
            self.shape = (int(self.shape[0]), int(self.shape[1]))
            assert self.scale_range[0] <= self.scale_range[1], (
                "Invalid scale range: "
                f"{self.scale_range[1]} < {self.scale_range[0]}"
            )

        self.interpolation = self.cfg.interpolation
        assert self.interpolation in ["nearest", "bilinear", "bicubic"]

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate current parameters."""
        params = super().generate_parameters(sample)
        if self.multiscale_mode == "range":
            assert isinstance(self.shape, tuple)
            if self.scale_range[0] < self.scale_range[1]:  # do multi-scale
                w_scale = (
                    torch.rand((len(sample),))
                    * (self.scale_range[1] - self.scale_range[0])
                    + self.scale_range[0]
                )
                h_scale = (
                    torch.rand((len(sample),))
                    * (self.scale_range[1] - self.scale_range[0])
                    + self.scale_range[0]
                )
            else:
                h_scale = w_scale = 1.0

            h_new = (
                torch.tensor(self.shape[0]).repeat(len(sample)) * h_scale
            ).int()
            w_new = (
                torch.tensor(self.shape[1]).repeat(len(sample)) * w_scale
            ).int()
            shape = torch.stack([h_new, w_new], -1)
        else:
            assert isinstance(self.shape, list)
            shape = torch.tensor(random.sample(self.shape, k=len(sample)))

        if self.keep_ratio:
            for i, sh in enumerate(shape):
                w, h = sample.images.image_sizes[i]
                sh[0] = sh[1] / (w / h)

        transform = (
            torch.eye(3, device=sample.device)
            .unsqueeze(0)
            .repeat(len(sample), 1, 1)
        )
        for i, sh in enumerate(shape):
            if params["apply"][i]:
                w, h = sample.images.image_sizes[i]
                transform[i, 0, 0] = sh[1] / w
                transform[i, 1, 1] = sh[0] / h

        params["shape"] = shape
        params["transform"] = transform
        return params

    def _apply_tensor(
        self, inputs: torch.Tensor, shape: torch.Tensor
    ) -> torch.Tensor:
        """Apply resize."""
        align_corners = None if self.interpolation == "nearest" else False
        output = F.interpolate(
            inputs,
            (shape[0], shape[1]),
            mode=self.interpolation,
            align_corners=align_corners,
        )
        return output

    def apply_intrinsics(
        self, intrinsics: Intrinsics, parameters: AugParams
    ) -> Intrinsics:
        """Transform intrinsic camera matrix according to augmentations."""
        return Intrinsics(
            torch.matmul(parameters["transform"], intrinsics.tensor)
        )

    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        all_ims = []
        for i, im in enumerate(images):
            if parameters["apply"][i]:
                im_t = self._apply_tensor(im.tensor, parameters["shape"][i])
                all_ims.append(Images(im_t, [(im_t.shape[3], im_t.shape[2])]))
            else:
                all_ims.append(im)

        if len(all_ims) == 1:
            return all_ims[0]
        return Images.cat(all_ims)

    def apply_box2d(
        self, boxes: List[Boxes2D], parameters: AugParams
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                box.boxes[:, :4] = transform_bbox(
                    parameters["transform"][i],
                    box.boxes[:, :4],
                )
        return boxes

    def apply_mask(
        self, masks: List[Masks], parameters: AugParams
    ) -> List[Masks]:
        """Apply augmentation to input mask."""
        interp = self.interpolation
        self.interpolation = "nearest"
        for i, mask in enumerate(masks):
            if len(mask) > 0 and parameters["apply"][i]:
                mask.masks = (
                    self._apply_tensor(
                        mask.masks.float().unsqueeze(1), parameters["shape"][i]
                    )
                    .squeeze(1)
                    .type(mask.masks.dtype)
                )
        self.interpolation = interp
        return masks


class RandomCropConfig(BaseAugmentationConfig):
    """Config for RandomCrop."""

    shape: Union[
        Tuple[float, float],
        Tuple[int, int],
        List[Tuple[float, float]],
        List[Tuple[int, int]],
    ]
    crop_type: str = "absolute"
    allow_empty_crop: bool = True
    recompute_boxes2d: bool = False


class RandomCrop(BaseAugmentation):
    """RandomCrop augmentation class."""

    def __init__(
        self,
        cfg: BaseAugmentationConfig,
    ) -> None:
        """Init function.

        Args:
            cfg: Augmentation config.
        """
        super().__init__(cfg)
        self.cfg: RandomCropConfig = RandomCropConfig(**cfg.dict())
        assert self.cfg.crop_type in [
            "absolute",
            "relative",
            "absolute_range",
            "relative_range",
        ], f"Unknown crop type {self.cfg.crop_type}."

        if self.cfg.crop_type == "absolute":
            assert isinstance(self.cfg.shape, tuple)
            assert self.cfg.shape[0] > 0 and self.cfg.shape[1] > 0
            self.shape: Tuple[int, int] = (
                int(self.cfg.shape[0]),
                int(self.cfg.shape[1]),
            )

        elif self.cfg.crop_type == "relative":
            assert isinstance(self.cfg.shape, tuple)
            assert 0 < self.cfg.shape[0] <= 1 and 0 < self.cfg.shape[1] <= 1
            self.scale: Tuple[float, float] = self.cfg.shape

        elif "range" in self.cfg.crop_type:
            assert isinstance(self.cfg.shape, list)
            assert len(self.cfg.shape) == 2
            assert self.cfg.shape[1][0] >= self.cfg.shape[0][0]
            assert self.cfg.shape[1][1] >= self.cfg.shape[0][1]

            if "absolute" in self.cfg.crop_type:
                for crop in self.cfg.shape:
                    assert crop[0] > 0 and crop[1] > 0
                self.shape_min: Tuple[int, int] = (
                    int(self.cfg.shape[0][0]),
                    int(self.cfg.shape[0][1]),
                )
                self.shape_max: Tuple[int, int] = (
                    int(self.cfg.shape[1][0]),
                    int(self.cfg.shape[1][1]),
                )
            else:
                for crop in self.cfg.shape:
                    assert 0 < crop[0] <= 1 and 0 < crop[1] <= 1
                self.scale_min: Tuple[float, float] = self.cfg.shape[0]
                self.scale_max: Tuple[float, float] = self.cfg.shape[1]

    def _get_crop_size(self, im_wh: torch.Tensor) -> Tuple[int, int]:
        """Generate random absolute crop size."""
        w, h = im_wh
        if self.cfg.crop_type == "absolute":
            return (
                min(int(self.shape[0]), h),
                min(int(self.shape[1]), w),
            )
        if self.cfg.crop_type == "absolute_range":
            crop_h = np.random.randint(
                min(h, self.shape_min[0]), min(h, self.shape_max[0]) + 1
            )
            crop_w = np.random.randint(
                min(w, self.shape_min[1]), min(w, self.shape_max[1]) + 1
            )
            return int(crop_h), int(crop_w)
        if self.cfg.crop_type == "relative":
            crop_h, crop_w = self.scale
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        # relative range
        crop_h = (
            np.random.rand() * (self.scale_max[0] - self.scale_min[0])
            + self.scale_min[0]
        )
        crop_w = (
            np.random.rand() * (self.scale_max[1] - self.scale_min[1])
            + self.scale_min[1]
        )
        return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def _sample_crop(self, im_wh: torch.Tensor) -> torch.Tensor:
        """Sample crop parameters according to config."""
        crop_size = self._get_crop_size(im_wh)
        margin_h = max(im_wh[1] - crop_size[0], 0)
        margin_w = max(im_wh[0] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        return torch.LongTensor([crop_x1, crop_y1, crop_x2, crop_y2])

    @staticmethod
    def _get_keep_mask(
        sample: InputSample, crop_param: torch.Tensor
    ) -> torch.Tensor:
        """Get mask for 2D annotations to keep."""
        assert len(sample) == 1, "Please provide a single sample!"
        assert len(crop_param.shape) == 1, "Please provide single crop_param"
        if len(sample.boxes2d[0]) > 0:
            # will be better to compute mask intersection (if exists) instead
            cropbox = Boxes2D(crop_param.float().unsqueeze(0))
            overlap = bbox_intersection(sample.boxes2d[0], cropbox)
            return overlap.squeeze(-1) > 0
        return torch.tensor([True] * len(sample.semantic_masks[0]))

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate current parameters."""
        parameters = super().generate_parameters(sample)
        image_whs = []
        crop_params = []
        keep_masks = []
        current_sample: InputSample
        for i, current_sample in enumerate(sample):
            im_wh = torch.tensor(current_sample.images.image_sizes[0])
            image_whs.append(im_wh)
            if not parameters["apply"][i]:
                crop_params.append(torch.tensor([0, 0, *im_wh]))
                num_objs = max(
                    len(current_sample.boxes2d),
                    len(current_sample.semantic_masks),
                )
                keep_masks.append(torch.tensor([True] * num_objs))
                continue

            crop_param = self._sample_crop(im_wh)
            keep_mask = self._get_keep_mask(current_sample, crop_param)
            while (
                len(current_sample.boxes2d[0]) > 0
                and not self.cfg.allow_empty_crop
                and keep_mask.sum() == 0
            ):  # pragma: no cover
                crop_param = self._sample_crop(im_wh)
                keep_mask = self._get_keep_mask(current_sample, crop_param)
            crop_params.append(crop_param)
            keep_masks.append(keep_mask)

        parameters["image_wh"] = torch.stack(image_whs)
        parameters["crop_params"] = torch.stack(crop_params)
        parameters["keep"] = keep_masks
        return parameters

    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        all_ims: List[Images] = []
        for i, im in enumerate(images):
            if parameters["apply"][i]:
                im_wh = im.image_sizes[0]
                x1, y1, x2, y2 = parameters["crop_params"][i]
                w, h = (x2 - x1).item(), (y2 - y1).item()
                im.tensor = im.tensor[:, :, y1:y2, x1:x2]
                im.image_sizes[i] = (min(im_wh[0], w), min(im_wh[1], h))
            all_ims.append(im)
        return Images.cat(all_ims)

    def apply_box2d(
        self,
        boxes: List[Boxes2D],
        parameters: AugParams,
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                offset = parameters["crop_params"][i, :2]
                box.boxes[:, :4] -= torch.cat([offset, offset])
                boxes[i] = box[parameters["keep"][i]]
        return boxes

    def apply_box3d(
        self,
        boxes: List[Boxes3D],
        parameters: AugParams,
    ) -> List[Boxes3D]:
        """Apply augmentation to input box3d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                boxes[i] = box[parameters["keep"][i]]
        return boxes

    def apply_intrinsics(
        self,
        intrinsics: Intrinsics,
        parameters: AugParams,
    ) -> Intrinsics:
        """Apply augmentation to input intrinsics."""
        x1, y1, _, _ = parameters["crop_params"].T
        intrinsics.tensor[:, 0, 2] -= x1
        intrinsics.tensor[:, 1, 2] -= y1
        return intrinsics

    def apply_mask(
        self,
        masks: List[Masks],
        parameters: AugParams,
    ) -> List[Masks]:
        """Apply augmentation to input mask."""
        for i, mask in enumerate(masks):
            if len(mask) > 0 and parameters["apply"][i]:
                x1, y1, x2, y2 = parameters["crop_params"][i]
                mask.masks = mask.masks[:, y1:y2, x1:x2]
                masks[i] = mask[parameters["keep"][i]]
        return masks

    def __call__(
        self, sample: InputSample, parameters: Optional[AugParams] = None
    ) -> Tuple[InputSample, AugParams]:
        """Apply augmentations to input sample."""
        # if parameters is given, still re-calculate keep / image_wh parameters
        if parameters is not None:
            parameters["image_wh"] = torch.stack(
                [torch.tensor(s.images.image_sizes[0]) for s in sample]
            )
            parameters["keep"] = [
                self._get_keep_mask(s, c)
                for s, c in zip(sample, parameters["crop_params"])
            ]
        sample, parameters = super().__call__(sample, parameters)
        if self.cfg.recompute_boxes2d:
            for i in range(len(sample)):
                assert len(sample.instance_masks[i]) == len(
                    sample.boxes2d[i]
                ), (
                    "recompute_boxes2d activated but annotations do not "
                    "contain instance masks!"
                )
                sample.boxes2d[i] = sample.instance_masks[i].get_boxes2d()
        return sample, parameters
