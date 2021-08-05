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

import kornia.augmentation as K
import torch
from detectron2.data.transforms import Augmentation, Transform
from detectron2.data.transforms import augmentation_impl as Augmentations
from kornia.augmentation.base import GeometricAugmentationBase2D
from pydantic.main import BaseModel

from openmt.common.registry import RegistryHolder


class AugmentationConfig(BaseModel):
    """Data augmentation instance config."""

    type: str
    kwargs: Dict[str, Union[bool, float, str, Tuple[int, int]]]


class BaseAugmentation(Augmentation, metaclass=RegistryHolder):  # type: ignore
    """Subclass detectron2 Augmentation to support registry."""

    def get_transform(
        self, *args: Dict[str, Union[bool, float, str, Tuple[int, int]]]
    ) -> Transform:
        """Get the corresponding deterministic transform for a given input.

        Args:
            args: Any fixed-length positional arguments. By default, the name
            of the arguments should exist in the :class:`AugInput` to be used.

        Returns:
            Transform: Returns the deterministic transform.
        """
        raise NotImplementedError


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


def build_augmentation_det2(
    cfg: AugmentationConfig,
) -> Union[Augmentation, Transform]:
    """Build a single detectron2 augmentation."""
    registry = RegistryHolder.get_registry(Augmentation)
    registry.update(RegistryHolder.get_registry(Transform))
    if hasattr(Augmentations, cfg.type):
        augmentation = getattr(Augmentations, cfg.type)
    elif cfg.type in registry:
        augmentation = registry[cfg.type]
    else:
        raise ValueError(f"Augmentation {cfg.type} not known!")
    return augmentation(**cfg.kwargs)


def build_augmentation(
    cfg: AugmentationConfig,
):
    """Build a single kornia augmentation."""
    registry = RegistryHolder.get_registry(GeometricAugmentationBase2D)

    if cfg.type in registry:
        augmentation = registry[cfg.type]
    elif hasattr(K, cfg.type):
        augmentation = getattr(K, cfg.type)
    else:
        raise ValueError(f"Augmentation {cfg.type} not known!")
    return augmentation(**cfg.kwargs)


def build_augmentations(
    cfgs: Optional[List[AugmentationConfig]],
) -> List[Union[Augmentation, Transform]]:
    """Build a list of augmentations / transforms and return these as List."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
