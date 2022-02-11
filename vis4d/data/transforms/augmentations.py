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
    LabelInstances,
    SemanticMasks,
    TMasks,
)

from .base import AugParams, BaseAugmentation
from .utils import get_resize_shape


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

        for i, sh in enumerate(shape):
            sh[1], sh[0] = get_resize_shape(
                sample.images.image_sizes[i], (sh[1], sh[0]), self.keep_ratio
            )

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
        self, boxes: List[Boxes2D], parameters: AugParams
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                offset = parameters["crop_params"][i, :2]
                box.boxes[:, :4] -= torch.cat([offset, offset])
                boxes[i] = box[parameters["keep"][i]]
        return boxes

    def apply_box3d(
        self, boxes: List[Boxes3D], parameters: AugParams
    ) -> List[Boxes3D]:
        """Apply augmentation to input box3d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                boxes[i] = box[parameters["keep"][i]]
        return boxes

    def apply_intrinsics(
        self, intrinsics: Intrinsics, parameters: AugParams
    ) -> Intrinsics:
        """Apply augmentation to input intrinsics."""
        x1, y1, _, _ = parameters["crop_params"].T
        intrinsics.tensor[:, 0, 2] -= x1
        intrinsics.tensor[:, 1, 2] -= y1
        return intrinsics

    def apply_instance_mask(
        self, masks: List[InstanceMasks], parameters: AugParams
    ) -> List[InstanceMasks]:
        """Apply augmentation to input instance mask."""
        for i, mask in enumerate(masks):
            if len(mask) > 0 and parameters["apply"][i]:
                x1, y1, x2, y2 = parameters["crop_params"][i]
                mask.masks = mask.masks[:, y1:y2, x1:x2]
                masks[i] = mask[parameters["keep"][i]]
        return masks

    def apply_semantic_mask(
        self, masks: List[SemanticMasks], parameters: AugParams
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
    """Mosaic augmentation.

    NOTE: So far, it only works with Images and Boxes2D. There is no support
    for other inputs or targets like InstanceMasks, etc.
    """

    def __init__(
        self,
        out_shape: Tuple[int, int],
        *args: ArgsType,
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        pad_value: float = 114.0,
        interpolation: str = "bilinear",
        clip_inside_image: bool = True,
        **kwargs: ArgsType,
    ) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        self.num_samples = 4
        self.out_shape = out_shape
        self.center_ratio_range = center_ratio_range
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.clip_inside_image = clip_inside_image

    def _mosaic_combine(
        self, index: int, center: Tuple[int, int], im_wh: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Compute the mosaic parameters for the image at the current index.

        Index:
        0 = top_left, 1 = top_right, 3 = bottom_left, 4 = bottom_right
        """
        assert index in (0, 1, 2, 3)
        if index == 0:
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center[0] - im_wh[0], 0),
                max(center[1] - im_wh[1], 0),
                center[0],
                center[1],
            )
            crop_coord = (
                im_wh[0] - (x2 - x1),
                im_wh[1] - (y2 - y1),
                im_wh[0],
                im_wh[1],
            )

        elif index == 1:
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center[0],
                max(center[1] - im_wh[1], 0),
                min(center[0] + im_wh[0], self.out_shape[1] * 2),
                center[1],
            )
            crop_coord = (
                0,
                im_wh[1] - (y2 - y1),
                min(im_wh[0], x2 - x1),
                im_wh[1],
            )

        elif index == 2:
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center[0] - im_wh[0], 0),
                center[1],
                center[0],
                min(self.out_shape[0] * 2, center[1] + im_wh[1]),
            )
            crop_coord = (
                im_wh[0] - (x2 - x1),
                0,
                im_wh[0],
                min(y2 - y1, im_wh[1]),
            )

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center[0],
                center[1],
                min(center[0] + im_wh[0], self.out_shape[1] * 2),
                min(self.out_shape[0] * 2, center[1] + im_wh[1]),
            )
            crop_coord = 0, 0, min(im_wh[0], x2 - x1), min(y2 - y1, im_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate parameters for mosaic."""
        assert len(sample) == self.num_samples, (
            "Number of images must be equal to the number of samples "
            "required for creating the mosaic."
        )
        h, w = self.out_shape
        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * w)
        center_y = int(random.uniform(*self.center_ratio_range) * h)
        center = (center_x, center_y)

        paste_coords, crop_coords, im_scales, im_shapes = [], [], [], []
        for i, img in enumerate(sample.images):
            # resize current image
            ori_wh = img.image_sizes[0]
            w_i, h_i = get_resize_shape(ori_wh, (w, h), keep_ratio=True)

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                i, center, (w_i, h_i)
            )
            paste_coords.append(paste_coord)
            crop_coords.append(crop_coord)
            im_shapes.append((w_i, h_i))
            im_scales.append((w_i / ori_wh[0], h_i / ori_wh[1]))

        parameters = dict(
            im_shapes=im_shapes,
            im_scales=im_scales,
            paste_params=paste_coords,
            crop_params=crop_coords,
        )
        return parameters

    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply augmentation to input image."""
        h, w = self.out_shape
        c = images.tensor.shape[1]
        mosaic_img = torch.full((1, c, h * 2, w * 2), self.pad_value)

        for i, img in enumerate(images):
            # resize current image
            w_i, h_i = parameters["im_shapes"][i]
            img.resize((h_i, w_i), self.interpolation)

            x1_p, y1_p, x2_p, y2_p = parameters["paste_params"][i]
            x1_c, y1_c, x2_c, y2_c = parameters["crop_params"][i]

            # crop and paste image
            mosaic_img[:, :, y1_p:y2_p, x1_p:x2_p] = img.tensor[
                :, :, y1_c:y2_c, x1_c:x2_c
            ]
        return Images(mosaic_img, image_sizes=[(w * 2, h * 2)])

    def apply_box2d(
        self, boxes: List[Boxes2D], parameters: AugParams
    ) -> List[Boxes2D]:
        """Apply augmentation to input box2d."""
        for i, box in enumerate(boxes):
            paste_coord, crop_coord, im_scale = (
                parameters["paste_params"][i],
                parameters["crop_params"][i],
                parameters["im_scales"][i],
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, _, _ = crop_coord

            # adjust boxes to new image size and origin coord
            if len(box) > 0:
                # add image prefix to track ids:
                if max(boxes[i].track_ids) >= 1000:
                    raise ValueError("Mosaic assumes < 1000 labels per image.")
                boxes[i].track_ids += 1000 * i

                pw = x1_p - x1_c
                ph = y1_p - y1_c
                box.boxes[:, [0, 2]] = im_scale[0] * box.boxes[:, [0, 2]] + pw
                box.boxes[:, [1, 3]] = im_scale[1] * box.boxes[:, [1, 3]] + ph

                # filter boxes outside current image
                crop_box = Boxes2D(torch.tensor(paste_coord).unsqueeze(0))
                overlap = bbox_intersection(box, crop_box)
                keep_mask = overlap.squeeze(-1) > 0
                boxes[i] = box[keep_mask]

                if self.clip_inside_image:
                    boxes[i].boxes[:, [0, 2]] = (
                        boxes[i].boxes[:, [0, 2]].clip(x1_p, x2_p)
                    )
                    boxes[i].boxes[:, [1, 3]] = (
                        boxes[i].boxes[:, [1, 3]].clip(y1_p, y2_p)
                    )

        return [Boxes2D.merge(boxes)]

    def __call__(
        self, sample: InputSample, parameters: Optional[AugParams] = None
    ) -> Tuple[InputSample, AugParams]:
        """Apply augmentations to input sample."""
        if parameters is None or not self.same_on_ref:
            parameters = self.generate_parameters(sample)  # pragma: no cover

        images = self.apply_image(sample.images, parameters)
        boxes2d = self.apply_box2d(sample.targets.boxes2d, parameters)

        new_sample = InputSample(
            [sample.metadata[0]],
            images,
            targets=LabelInstances(boxes2d=boxes2d),
        )
        return new_sample, parameters


class MixUp(BaseAugmentation):
    """MixUp Augmentation.

                        mixup transform
               +------------------------------+
               | mixup image   |              |
               |      +--------|--------+     |
               |      |        |        |     |
               |---------------+        |     |
               |      |                 |     |
               |      |      image      |     |
               |      |                 |     |
               |      |                 |     |
               |      |-----------------+     |
               |             pad              |
               +------------------------------+

    The mixup transform steps are as follows::
       1. Another random image is picked by dataset and embedded in
          the top left patch(after padding and resizing)
       2. The target of mixup transform is the weighted average of mixup
          image and origin image.

    NOTE: So far, it only works with Images and Boxes2D. There is no support
    for other inputs or targets like InstanceMasks, etc.
    """

    def __init__(
        self,
        *args: ArgsType,
        out_shape: Tuple[int, int],
        flip_ratio: float = 0.5,
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        clip_inside_image: bool = True,
        pad_value: float = 114.0,
        interpolation: str = "bilinear",
        **kwargs: ArgsType,
    ) -> None:
        """Init function."""
        super().__init__(*args, **kwargs)
        self.num_samples = 2
        self.out_shape = out_shape
        self.flip_ratio = flip_ratio
        self.ratio_range = ratio_range
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.clip_inside_image = clip_inside_image

    def generate_parameters(self, sample: InputSample) -> AugParams:
        """Generate parameters for MixUp."""
        assert len(sample) == 2, "MixUp only supports num_samples=2!"
        h, w = self.out_shape
        ori_img, other_img = sample.images[0], sample.images[1]
        ori_w, ori_h = ori_img.image_sizes[0]

        is_flip = random.uniform(0, 1) > self.flip_ratio
        other_ori_wh = other_img.image_sizes[0]
        w_i, h_i = get_resize_shape(other_ori_wh, (w, h), keep_ratio=True)
        jit_factor = random.uniform(*self.ratio_range)
        h_i, w_i = int(jit_factor * h_i), int(jit_factor * w_i)
        pad_shape = (max(h_i, ori_h), max(w_i, ori_w))

        x_offset, y_offset = 0, 0
        if pad_shape[0] > ori_h:
            y_offset = random.randint(0, pad_shape[0] - ori_h)
        if pad_shape[1] > ori_w:
            x_offset = random.randint(0, pad_shape[1] - ori_w)

        parameters = dict(
            im_scale=(w_i / other_ori_wh[0], h_i / other_ori_wh[1]),
            im_shape=(w_i, h_i),
            other_ori_wh=other_ori_wh,
            other_new_wh=(min(w_i, ori_w), min(h_i, ori_h)),
            pad_hw=pad_shape,
            is_flip=is_flip,
            crop_coord=(
                x_offset,
                y_offset,
                x_offset + ori_w,
                y_offset + ori_h,
            ),
        )
        return parameters

    def apply_image(self, images: Images, parameters: AugParams) -> Images:
        """Apply MixUp on image pair."""
        h, w = self.out_shape
        c = images.tensor.shape[1]
        ori_img, other_img = images[0], images[1]
        w_i, h_i = parameters["im_shape"]

        # resize, scale jitter other image
        other_img.resize((h_i, w_i), self.interpolation)

        # random horizontal flip other image
        if parameters["is_flip"]:
            other_img.flip()

        # pad, optionally random crop other image
        padded_img = torch.full((1, c, *parameters["pad_hw"]), self.pad_value)
        padded_img[:, :, :h_i, :w_i] = other_img[0].tensor
        x1_c, y1_c, x2_c, y2_c = parameters["crop_coord"]
        padded_cropped_img = padded_img[:, :, y1_c:y2_c, x1_c:x2_c]

        # mix ori and other
        mixup_img = 0.5 * ori_img[0].tensor + 0.5 * padded_cropped_img
        return Images(mixup_img, image_sizes=[(w * 2, h * 2)])

    def apply_box2d(
        self, boxes: List[Boxes2D], parameters: AugParams
    ) -> List[Boxes2D]:
        """Apply MixUp to Boxes2D."""
        assert len(boxes) == 2, "MixUp only supports num_samples=2!"
        ori_boxes, other_boxes = boxes[0], boxes[1]

        crop_coord, im_scale = parameters["crop_coord"], parameters["im_scale"]
        x1_c, y1_c, _, _ = crop_coord

        # adjust boxes to new image size and origin coord
        if len(other_boxes) > 0:
            if parameters["is_flip"]:
                w = parameters["other_ori_wh"][0]
                other_boxes.boxes[:, [0, 2]] = w - other_boxes.boxes[:, [0, 2]]

            other_boxes.boxes[:, [0, 2]] = (
                im_scale[0] * other_boxes.boxes[:, [0, 2]] - x1_c
            )
            other_boxes.boxes[:, [1, 3]] = (
                im_scale[1] * other_boxes.boxes[:, [1, 3]] - y1_c
            )

            if (
                max(other_boxes.track_ids) >= 1000
                or max(ori_boxes.track_ids) >= 1000
            ):
                raise ValueError("MixUp assumes < 1000 labels per image")
            other_boxes.track_ids += 1000

            # filter boxes outside other image
            crop_box = Boxes2D(torch.tensor(crop_coord).unsqueeze(0))
            overlap = bbox_intersection(other_boxes, crop_box)
            keep_mask = overlap.squeeze(-1) > 0
            other_boxes = other_boxes[keep_mask]

            if self.clip_inside_image:
                new_w, new_h = parameters["other_new_wh"]
                other_boxes.boxes[:, [0, 2]] = other_boxes.boxes[
                    :, [0, 2]
                ].clip(0, new_w)
                other_boxes.boxes[:, [1, 3]] = other_boxes.boxes[
                    :, [1, 3]
                ].clip(0, new_h)

        return [Boxes2D.merge([ori_boxes, other_boxes])]

    def __call__(
        self, sample: InputSample, parameters: Optional[AugParams] = None
    ) -> Tuple[InputSample, AugParams]:
        """Apply augmentations to input sample."""
        if parameters is None or not self.same_on_ref:
            parameters = self.generate_parameters(sample)  # pragma: no cover

        images = self.apply_image(sample.images, parameters)
        boxes2d = self.apply_box2d(sample.targets.boxes2d, parameters)

        new_sample = InputSample(
            [sample.metadata[0]],
            images,
            targets=LabelInstances(boxes2d=boxes2d),
        )
        return new_sample, parameters
