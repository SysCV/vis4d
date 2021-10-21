"""Interface for using Kornia augmentations in VisT.

Kornia augmentations use the following functions:
- generate_parameters: Generate the (random) parameters for computing the
current transformation.
- compute_transformation: Compute a geometric transformation that transforms
the input. Optionally in Kornia but required here, since we need the transform
to maintain valid projective geometry in 3D tracking.
- apply_transform: Apply the transformation to the input.

Reference: https://kornia.readthedocs.io/en/latest/augmentation.base.html
"""
from typing import Sequence, Optional, Union, Dict, Tuple, List

import torch

from kornia import augmentation as kornia_augmentation
from vist.data.utils import transform_bbox
from vist.struct import DictStrAny, Images
from vist.struct.labels import Bitmasks

from .base import BaseAugmentation, BaseAugmentationConfig


class KorniaAugmentationConfig(BaseAugmentationConfig):
    """Config for Kornia augmentation wrapper."""

    kornia_type: Optional[str] = None
    kwargs: Dict[
        str,
        Union[
            bool,
            float,
            str,
            Tuple[float, float],
            Tuple[int, int],
            List[Tuple[int, int]],
        ],
    ]


class KorniaAugmentationWrapper(BaseAugmentation):
    """Kornia augmentation wrapper class."""

    def __init__(self, cfg: BaseAugmentationConfig):
        """Initialize wrapper."""
        super().__init__(cfg)
        self.cfg: KorniaAugmentationConfig = KorniaAugmentationConfig(**cfg.dict())
        self.prob = 1.0
        augmentation = getattr(
            kornia_augmentation, cfg.kornia_type  # type: ignore
        )
        self.augmentor = augmentation(**cfg.kwargs)

    def generate_parameters(self, batch_shape: torch.Size) -> DictStrAny:
        """Generate current parameters."""
        parameters = super().generate_parameters(batch_shape)
        to_apply = parameters["batch_prob"]
        _params = self.augmentor.generate_parameters(
            torch.Size((int(to_apply.sum().item()), *batch_shape[1:]))
        )
        parameters.update(_params)
        return parameters

    def apply_intrinsics(
        self,
        intrinsics: Intrinsics,
        parameters: DictStrAny,
    ) -> Intrinsics:
        """Apply augmentation to input intrinsics."""
        self.augmentor.compute_transformation(inputs, parameters)
        return intrinsics

    def apply_transform(
        self,
        inputs: torch.Tensor,
        params: AugParams,
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the transformation given parameters and transform."""
        return self.augmentor.apply_transform(inputs, params, transform)

    def apply_image(
        self, image: Images, parameters: DictStrAny, transform: torch.Tensor
    ) -> Images:
        """Apply augmentation to input image."""
        imaget = self.apply_transform(image.tensor, parameters, transform)
        return Images(imaget, [(imaget.shape[3], imaget.shape[2])])

    def apply_box2d(  # pylint: disable=unused-argument
        self,
        boxes: Sequence[Boxes2D],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Boxes2D]:
        """Apply augmentation to input box2d."""
        if len(boxes[0]) != 0:
            boxes[0].boxes[:, :4] = transform_bbox(
                transform[0],
                boxes[0].boxes[:, :4],
            )
        return boxes

    def apply_box3d(  # pylint: disable=unused-argument
        self,
        boxes: Sequence[Boxes3D],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Boxes3D]:
        """Apply augmentation to input box3d."""
        return boxes

    def apply_mask(
        self,
        masks: Sequence[Bitmasks],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Bitmasks]:
        """Apply augmentation to input mask."""
        if len(masks[0]) != 0:
            masks[0].masks = (
                self.apply_transform(
                    masks[0].masks.float().unsqueeze(1), parameters, transform
                )
                .squeeze(1)
                .type(masks[0].masks.dtype)
            )
        return masks

    def __call__(
        self, sample: InputSample, parameters: DictStrAny, training: bool
    ) -> Tuple[InputSample, torch.Tensor]:
        """Apply augmentations to input sample."""
        bprob = parameters["batch_prob"]
        # if no augmentation needed
        if torch.sum(bprob) == 0:
            trans_matrix = [identity_matrix(sample.images.tensor)]
        # if all data needs to be augmented
        elif torch.sum(bprob) == len(bprob):
            _, trans_matrix = super().__call__(sample, parameters, training)
        else:
            trans_matrix = identity_matrix(sample.images.tensor)
            _, trans_matrix[bprob] = super().__call__(
                sample[bprob], parameters, training
            )

        return sample, trans_matrix

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        return self.augmentor.__repr__()  # type: ignore


class KorniaColorJitter(KorniaAugmentationWrapper):
    """Wrapper for Kornia color jitter augmentation class."""

    def apply_image(
        self, image: Images, parameters: DictStrAny, transform: torch.Tensor
    ) -> Images:
        """Apply augmentation to input image."""
        imaget = self.apply_transform(
            image.tensor / 255.0, parameters, transform
        )
        return Images(
            (imaget * 255).type(image.tensor.dtype),
            [(imaget.shape[3], imaget.shape[2])],
        )

    def apply_mask(
        self,
        masks: Sequence[Bitmasks],
        parameters: DictStrAny,
        transform: torch.Tensor,
    ) -> Sequence[Bitmasks]:
        """Skip augmentation for mask."""
        return masks
