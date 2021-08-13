"""VisT augmentations."""
import random
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from .base import AugParams, BaseAugmentation


class Resize(BaseAugmentation):
    """Simple resize augmentation class."""

    def __init__(
        self,
        shape: Union[Tuple[int, int], List[Tuple[int, int]]],
        keep_ratio: bool = False,
        multiscale_mode: str = "range",
        scale_range: Tuple[float, float] = (1.0, 1.0),
        interpolation: str = "bilinear",
    ) -> None:
        """Init function.

        Args:
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
        super().__init__(p=1.0)
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
            self.shape = (
                [  # pylint: disable=unsubscriptable-object,line-too-long
                    (int(s[0]), int(s[1])) for s in self.shape
                ]
            )
        else:
            assert isinstance(
                self.shape, tuple
            ), "Specify shape as tuple when using multiscale mode range."
            self.shape = (int(self.shape[0]), int(self.shape[1]))
            assert (
                scale_range[0] <= scale_range[1]
            ), f"Invalid scale range: {scale_range[1]} < {scale_range[0]}"

        self.interpolation = interpolation
        assert self.interpolation in ["nearest", "bilinear", "bicubic"]

    def generate_parameters(self, batch_shape: torch.Size) -> AugParams:
        """Generate current parameters."""
        if self.multiscale_mode == "range":
            assert isinstance(self.shape, tuple)
            if self.scale_range[0] < self.scale_range[1]:  # do multi-scale
                w_scale = (
                    random.random()
                    * (self.scale_range[1] - self.scale_range[0])
                    + self.scale_range[0]
                )
                h_scale = (
                    random.random()
                    * (self.scale_range[1] - self.scale_range[0])
                    + self.scale_range[0]
                )
                shape = (
                    int(self.shape[0] * h_scale),
                    int(self.shape[1] * w_scale),
                )
            else:
                shape = self.shape
        else:
            assert isinstance(self.shape, list)
            shape = random.sample(self.shape, k=1)[0]

        if self.keep_ratio:
            _, _, h, w = batch_shape
            shape = (int(shape[1] / (w / h)), shape[1])

        return dict(shape=shape)

    def compute_transformation(
        self, inputs: torch.Tensor, params: AugParams
    ) -> torch.Tensor:
        """Compute transformation for resize."""
        transform = torch.eye(3, device=inputs.device)
        n, _, h, w = inputs.shape
        transform[0, 0] = params["shape"][1] / w
        transform[1, 1] = params["shape"][0] / h
        return torch.stack([transform for _ in range(n)], 0)

    def apply_transform(
        self, inputs: torch.Tensor, params: AugParams, transform: torch.Tensor
    ) -> torch.Tensor:
        """Apply resize."""
        align_corners = None if self.interpolation == "nearest" else False
        output = F.interpolate(
            inputs,
            params["shape"],
            mode=self.interpolation,
            align_corners=align_corners,
        )
        return output
