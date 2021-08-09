"""Interface for using and extending detectron2 augmentations in openMT.

Detectron2 spans two types of operations:
- Augmentation defines the “policy” to modify inputs.
- Transform implements the actual operations to transform data
Depending on the use case, you may want to add either the Augmentation or the
Transform type to your AugmentationList (e.g. probabilistic vs. deterministic
execution).

Reference: https://detectron2.readthedocs.io/tutorials/augmentation.html
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from kornia import augmentation as kornia_augmentation
from kornia.augmentation.base import GeometricAugmentationBase2D
from pydantic.main import BaseModel

from openmt.common.registry import RegistryHolder


class AugmentationConfig(BaseModel):
    """Data augmentation instance config."""

    type: str
    kwargs: Dict[str, Union[bool, float, str, Tuple[int, int]]]


class BaseKorniaAugmentation(
    GeometricAugmentationBase2D, metaclass=RegistryHolder
):
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
):
    """Build a single kornia augmentation."""
    registry = RegistryHolder.get_registry(GeometricAugmentationBase2D)

    if cfg.type in registry:
        augmentation = registry[cfg.type]
    elif hasattr(kornia_augmentation, cfg.type):
        augmentation = getattr(kornia_augmentation, cfg.type)
    else:
        raise ValueError(f"Augmentation {cfg.type} not known!")
    return augmentation(**cfg.kwargs)


def build_augmentations(
    cfgs: Optional[List[AugmentationConfig]],
):
    """Build a list of augmentations / transforms and return these as List."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
