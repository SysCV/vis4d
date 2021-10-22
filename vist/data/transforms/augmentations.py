"""VisT augmentations."""
import random
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from vist.data.utils import transform_bbox
from vist.struct import Boxes2D, Images, InputSample, Intrinsics, Masks

from .base import AugParams, BaseAugmentation, BaseAugmentationConfig


class ResizeAugmentationConfig(BaseAugmentationConfig):
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
        self.cfg: ResizeAugmentationConfig = ResizeAugmentationConfig(
            **cfg.dict()
        )
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
            torch.eye(3, device=sample.images.device)
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
        for i, im in enumerate(images):  # type: ignore
            im: Images  # type: ignore
            if parameters["apply"][i]:
                im_t = self._apply_tensor(im.tensor, parameters["shape"][i])
                all_ims.append(Images(im_t, [(im_t.shape[3], im_t.shape[2])]))
            else:
                all_ims.append(im)

        if len(all_ims) == 1:
            return all_ims[0]
        return Images.cat(all_ims)

    def apply_box2d(
        self, boxes: Sequence[Boxes2D], parameters: AugParams
    ) -> Sequence[Boxes2D]:
        """Apply augmentation to input box2d."""
        for i, box in enumerate(boxes):
            if len(box) > 0 and parameters["apply"][i]:
                box.boxes[:, :4] = transform_bbox(
                    parameters["transform"][i],
                    box.boxes[:, :4],
                )
        return boxes

    def apply_mask(
        self, masks: Sequence[Masks], parameters: AugParams
    ) -> Sequence[Masks]:
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
