"""VisT augmentations."""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import AugParams, BaseAugmentation


class Resize(BaseAugmentation):
    """Simple resize augmentation class."""

    def __init__(
        self,
        shape: Tuple[int, int],
        interpolation: Optional[str] = None,
    ) -> None:
        """Init function.

        Args:
            shape: Image shape to be resized to in (H, W) format.
            interpolation: Interpolation method. One of ["nearest", "bilinear",
            "bicubic"]
            return_transform: If the transform should be returned in matrix
            format.
        """
        super().__init__(p=1.0)
        self.shape = shape
        if interpolation is None:
            self.interpolation = "bilinear"
        else:
            self.interpolation = interpolation

        assert self.interpolation in ["nearest", "bilinear", "bicubic"]

    def generate_parameters(self, batch_shape: torch.Size) -> AugParams:
        """Generate current parameters."""
        return dict(shape=self.shape)

    def compute_transformation(
        self, inputs: torch.Tensor, params: AugParams
    ) -> torch.Tensor:
        """Compute transformation for resize."""
        transform = torch.eye(3, 3, device=inputs.device)
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
            self.shape,
            mode=self.interpolation,
            align_corners=align_corners,
        )
        return output
