"""Interface for using and extending kornia augmentations in VisT.

Kornia augmentations require the following functions:
- generate_parameters: Generate the (random) parameters for computing the
current transformation.
- compute_transformation: Compute a geometric transformation that transforms
the input. Optionally in Kornia but required here, since we need the transform
to maintain valid projective geometry in 3D tracking.
- apply_transform: Apply the transformation to the input.

Reference: https://kornia.readthedocs.io/en/latest/augmentation.base.html
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from kornia import augmentation as kornia_augmentation
from kornia.augmentation.base import GeometricAugmentationBase2D
from pydantic.main import BaseModel

from vist.common.registry import RegistryHolder

AugParams = Dict[str, torch.Tensor]


class AugmentationConfig(BaseModel):
    """Data augmentation instance config."""

    type: str
    kwargs: Dict[str, Union[bool, float, str, Tuple[int, int]]]


class BaseAugmentation(GeometricAugmentationBase2D, metaclass=RegistryHolder):  # type: ignore # pylint: disable=line-too-long
    """Subclass kornia Augmentation to support registry."""

    def compute_transformation(  # pylint: disable=arguments-renamed
        self, inputs: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get the corresponding deterministic transform for a given input.

        Args:
            inputs: Image tensor.
            params: parameters needed for computing transformation matrix.

        Returns:
            Transform: Returns the deterministic transform.
        """
        raise NotImplementedError


def build_augmentation(
    cfg: AugmentationConfig,
) -> BaseAugmentation:
    """Build a single augmentation."""
    registry = RegistryHolder.get_registry(GeometricAugmentationBase2D)

    if cfg.type in registry:
        augmentation = registry[cfg.type]
    elif hasattr(kornia_augmentation, cfg.type):
        augmentation = getattr(kornia_augmentation, cfg.type)
    else:
        raise ValueError(f"Augmentation {cfg.type} not known!")
    return augmentation(**cfg.kwargs)  # type: ignore


def build_augmentations(
    cfgs: Optional[List[AugmentationConfig]],
) -> List[BaseAugmentation]:
    """Build a list of augmentations and return these as List."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
