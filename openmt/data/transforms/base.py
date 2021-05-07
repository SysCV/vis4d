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

from detectron2.data.transforms import Augmentation, Transform
from detectron2.data.transforms import augmentation_impl as Augmentations

from openmt.common.registry import RegistryHolder
from openmt.config import Augmentation as AugmentationConfig


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


def build_augmentation(
    cfg: AugmentationConfig,
) -> Union[Augmentation, Transform]:
    """Build a single detectron2 augmentation."""
    registry = RegistryHolder.get_registry(__package__)
    if hasattr(Augmentations, cfg.type):
        augmentation = getattr(Augmentations, cfg.type)
    elif cfg.type in registry:
        augmentation = registry[cfg.type]
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
