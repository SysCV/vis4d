"""Vis4D augmentations."""
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.distributed import rank_zero_warn

from vis4d.common.bbox.utils import bbox_intersection
from vis4d.data.utils import transform_bbox
from vis4d.struct import (
    ArgsType,
    Boxes2D,
    Boxes3D,
    Images,
    InputSample,
    InstanceMasks,
    Intrinsics,
    SemanticMasks,
    TMasks,
)

from .base import AugParams, BaseAugmentation


def im_resize():
    align_corners = None if self.interpolation == "nearest" else False
    im_resized = F.interpolate(
        inputs,
        (H, W),
        mode=self.interpolation,
        align_corners=align_corners,
    )

class Resize(BaseAugmentation):
    """Resize augmentation.

    Attributes:
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

    def __init__(
        self,
        shape: Union[Tuple[int, int], List[Tuple[int, int]]],
        *args: ArgsType,
        keep_ratio: bool = False,
        multiscale_mode: str = "range",
        scale_range: Tuple[float, float] = (1.0, 1.0),
        interpolation: str = "bilinear",
        **kwargs: ArgsType,
    ) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.keep_ratio = keep_ratio
        self.multiscale_mode = multiscale_mode
        assert self.multiscale_mode in ["list", "range"]
        self.scale_range = scale_range
        if self.multiscale_mode == "list":
            assert isinstance(
                self.shape, list
            ), "Specify shape as list when using multiscale mode list."
            assert len(self.shape) >= 1
            # pylint: disable=unsubscriptable-object
            self.shape = [(int(s[0]), int(s[1])) for s in self.shape]
        else:
            if (
                isinstance(self.shape, list)
                and isinstance(self.shape[0], int)
                and isinstance(self.shape[1], int)
            ):
                self.shape = tuple(self.shape)
            assert isinstance(
                self.shape, tuple
            ), "Specify shape as tuple when using multiscale mode range."
            self.shape = (int(self.shape[0]), int(self.shape[1]))
            assert self.scale_range[0] <= self.scale_range[1], (
                "Invalid scale range: "
                f"{self.scale_range[1]} < {self.scale_range[0]}"
            )

        self.interpolation = interpolation
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
                long_edge, short_edge = max(sh), min(sh)
                scale_factor = min(
                    long_edge / max(h, w), short_edge / min(h, w)
                )
                sh[0] = int(h * scale_factor + 0.5)
                sh[1] = int(w * scale_factor + 0.5)
        else:
            # if h is long edge in original image, but is not in the current
            # resize shape, we flip (h, w) to avoid large image distortions
            for i, sh in enumerate(shape):
                w, h = sample.images.image_sizes[i]
                if w < h and not sh[1] < sh[0]:
                    shape[i] = torch.flip(sh.unsqueeze(0), (0, 1)).squeeze(0)

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
        self, masks: List[TMasks], parameters: AugParams
    ) -> List[TMasks]:
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


class RandomCrop(BaseAugmentation):
    """RandomCrop augmentation class."""

    def __init__(
        self,
        shape: Union[
            Tuple[float, float],
            Tuple[int, int],
            List[Tuple[float, float]],
            List[Tuple[int, int]],
        ],
        *args: ArgsType,
        crop_type: str = "absolute",
        allow_empty_crops: bool = True,
        recompute_boxes2d: bool = False,
        cat_max_ratio: float = 1.0,
        **kwargs: ArgsType,
    ) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        assert crop_type in [
            "absolute",
            "relative",
            "absolute_range",
            "relative_range",
        ], f"Unknown crop type {crop_type}."
        self.crop_type = crop_type
        self.cat_max_ratio = cat_max_ratio
        self.allow_empty_crops = allow_empty_crops
        self.recompute_boxes2d = recompute_boxes2d
        if isinstance(shape, list) and (
            (isinstance(shape[0], int) and isinstance(shape[1], int))
            or (isinstance(shape[0], float) and isinstance(shape[1], float))
        ):
            shape = tuple(shape)

        if crop_type == "absolute":
            assert isinstance(shape, tuple)
            assert shape[0] > 0 and shape[1] > 0
            self.shape: Tuple[int, int] = (
                int(shape[0]),
                int(shape[1]),
            )

        elif crop_type == "relative":
            assert isinstance(shape, tuple)
            assert 0 < shape[0] <= 1 and 0 < shape[1] <= 1
            self.scale: Tuple[float, float] = shape

        elif "range" in crop_type:
            assert isinstance(shape, list)
            assert len(shape) == 2
            assert shape[1][0] >= shape[0][0]
            assert shape[1][1] >= shape[0][1]

            if "absolute" in crop_type:
                for crop in shape:
                    assert crop[0] > 0 and crop[1] > 0
                self.shape_min: Tuple[int, int] = (
                    int(shape[0][0]),
                    int(shape[0][1]),
                )
                self.shape_max: Tuple[int, int] = (
                    int(shape[1][0]),
                    int(shape[1][1]),
                )
            else:
                for crop in shape:
                    assert 0 < crop[0] <= 1 and 0 < crop[1] <= 1
                self.scale_min: Tuple[float, float] = shape[0]
                self.scale_max: Tuple[float, float] = shape[1]

    def _get_crop_size(self, im_wh: torch.Tensor) -> Tuple[int, int]:
        """Generate random absolute crop size."""
        w, h = im_wh
        if self.crop_type == "absolute":
            return (
                min(int(self.shape[0]), h),
                min(int(self.shape[1]), w),
            )
        if self.crop_type == "absolute_range":
            crop_h = np.random.randint(
                min(h, self.shape_min[0]), min(h, self.shape_max[0]) + 1
            )
            crop_w = np.random.randint(
                min(w, self.shape_min[1]), min(w, self.shape_max[1]) + 1
            )
            return int(crop_h), int(crop_w)
        if self.crop_type == "relative":
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
        crop_box = Boxes2D(crop_param.float().unsqueeze(0))
        if len(sample.targets.boxes2d[0]) > 0:
            # will be better to compute mask intersection (if exists) instead
            overlap = bbox_intersection(sample.targets.boxes2d[0], crop_box)
            return overlap.squeeze(-1) > 0
        return torch.tensor([])

    def _check_seg_max_cat(
        self, sample: InputSample, crop_param: torch.Tensor
    ) -> bool:
        """Check if any category occupies more than cat_max_ratio."""
        crop_box = Boxes2D(crop_param.float().unsqueeze(0))
        crop_masks = sample.targets.semantic_masks[0].crop(crop_box)
        cls_ids, cnts = torch.unique(
            crop_masks.to_hwc_mask(), return_counts=True
        )
        cnts = cnts[cls_ids != 255]
        keep_mask = (
            len(cnts) > 1 and cnts.max() / cnts.sum() < self.cat_max_ratio
        )
        return keep_mask

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate current parameters."""
        parameters = super().generate_parameters(sample)
        image_whs = []
        crop_params = []
        keep_masks = []
        cur_sample: InputSample
        for i, cur_sample in enumerate(sample):
            im_wh = torch.tensor(cur_sample.images.image_sizes[0])
            image_whs.append(im_wh)
            if not parameters["apply"][i]:
                crop_params.append(torch.tensor([0, 0, *im_wh]))
                num_objs = len(cur_sample.targets.boxes2d[0])
                keep_masks.append(torch.tensor([True] * num_objs))
                continue

            crop_param = self._sample_crop(im_wh)
            keep_mask = self._get_keep_mask(cur_sample, crop_param)
            if (
                len(cur_sample.targets.boxes2d[0]) > 0
                or self.cat_max_ratio != 1.0
            ):
                # resample crop if conditions not satisfied
                found_crop = False
                for _ in range(10):
                    # try resampling 10 times, otherwise use last crop
                    if (
                        self.allow_empty_crops or keep_mask.sum() != 0
                    ) and not self._check_seg_max_cat(cur_sample, crop_param):
                        found_crop = True
                        break
                    crop_param = self._sample_crop(im_wh)
                    keep_mask = self._get_keep_mask(cur_sample, crop_param)
                if not found_crop:
                    rank_zero_warn(
                        "Random crop not found within 10 resamples."
                    )
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
        if len(all_ims) == 1:
            return all_ims[0]
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

    def apply_instance_mask(
        self,
        masks: List[InstanceMasks],
        parameters: AugParams,
    ) -> List[InstanceMasks]:
        """Apply augmentation to input instance mask."""
        for i, mask in enumerate(masks):
            if len(mask) > 0 and parameters["apply"][i]:
                x1, y1, x2, y2 = parameters["crop_params"][i]
                mask.masks = mask.masks[:, y1:y2, x1:x2]
                masks[i] = mask[parameters["keep"][i]]
        return masks

    def apply_semantic_mask(
        self,
        masks: List[SemanticMasks],
        parameters: AugParams,
    ) -> List[SemanticMasks]:
        """Apply augmentation to input semantic mask."""
        for i, mask in enumerate(masks):
            if len(mask) > 0 and parameters["apply"][i]:
                x1, y1, x2, y2 = parameters["crop_params"][i]
                mask.masks = mask.masks[:, y1:y2, x1:x2]
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
        if self.recompute_boxes2d:
            for i in range(len(sample)):
                assert len(sample.targets.instance_masks[i]) == len(
                    sample.targets.boxes2d[i]
                ), (
                    "recompute_boxes2d activated but annotations do not "
                    "contain instance masks!"
                )
                sample.targets.boxes2d[i] = sample.targets.instance_masks[
                    i
                ].get_boxes2d()
        return sample, parameters


class Mosaic(BaseAugmentation):
    """Mosaic augmentation."""

    def __init__(
        self,
        num_samples: int,
        *args: ArgsType,
        **kwargs: ArgsType,
    ) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        self.num_samples = 4
        self.out_shape = (3, 800, 1440)

    def _mosaic_combine(self, index: int, center: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """Compute the mosaic parameters for the image at the current index.

        Index:
        0 = top_left, 1 = top_right, bottom_left, bottom_right
        """


    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        assert len(images) == self.num_samples, \
            "Number of images must be equal to the number of samples " \
            "required for creating the mosaic."
        C, H, W = self.out_shape
        mosaic_img = torch.zeros((1, C, H, W))

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * W)
        center_y = int(random.uniform(*self.center_ratio_range) * H)
        center = (center_x, center_y)

        for i, img in images:
            w_i, h_i = img.image_sizes[0]

            # resize current image
            # TODO keep ratio option

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(i, center, )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img[y1_c:y2_c, x1_c:x2_c]

        return images

    def apply_box2d(
        self, boxes: List[Boxes2D], parameters: AugParams
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        return boxes

    def __call__(
        self, sample: InputSample, parameters: Optional[AugParams] = None
    ) -> Tuple[InputSample, AugParams]:
        """Apply augmentations to input sample."""
        if parameters is None or not self.same_on_ref:
            parameters = self.generate_parameters(sample)

        images = self.apply_image(sample.images, parameters)
        boxes2d = self.apply_box2d(
            sample.targets.boxes2d, parameters
        )

        # TODO check for other targets / inputs, raise Exception

        new_sample = InputSample(sample.metadata[0], images, targets=LabelInstances(boxes2d=boxes2d))
        return new_sample
