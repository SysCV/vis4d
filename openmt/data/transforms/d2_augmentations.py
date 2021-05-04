"""Interface for using detectron2 augmentations in openMT.

Detectron2 spans two types of operations:
- Augmentation defines the “policy” to modify inputs.
- Transform implements the actual operations to transform data
Depending on the use case, you may want to add either the Augmentation or the
Transform type to your AugmentationList (e.g. probabilistic vs. deterministic
execution).

Reference: https://detectron2.readthedocs.io/tutorials/augmentation.html
"""
from typing import List, Optional, Union

from detectron2.data.transforms import Augmentation, Transform
from detectron2.data.transforms import augmentation_impl as Augmentations
from detectron2.data.transforms import transform as Transforms

from openmt.config import Augmentation as AugmentationConfig


def build_augmentation(
    cfg: AugmentationConfig,
) -> Union[Augmentation, Transform]:
    """Build a single detectron2 augmentation."""
    if hasattr(Augmentations, cfg.type):
        augmentation = getattr(Augmentations, cfg.type)
    elif hasattr(Transforms, cfg.type):
        augmentation = getattr(Transforms, cfg.type)
    else:
        raise ValueError(f"Augmentation {cfg.type} not known!")
    return augmentation(**cfg.kwargs)


def build_augmentations(
    cfgs: Optional[List[AugmentationConfig]],
) -> List[Union[Augmentation, Transform]]:
    """Build a list of augmentations and return these as AUgmentationList."""
    augmentations = []
    if cfgs is not None:
        for aug_cfg in cfgs:
            augmentations.append(build_augmentation(aug_cfg))
    return augmentations
