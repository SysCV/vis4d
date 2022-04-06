"""Standard data augmentation pipelines."""
from typing import List, Optional, Tuple

from vis4d.data.transforms import (
    BaseAugmentation,
    KorniaAugmentationWrapper,
    KorniaColorJitter,
    KorniaRandomHorizontalFlip,
    MixUp,
    Mosaic,
    RandomCrop,
    Resize,
)


def default(im_hw: Tuple[int, int]) -> List[BaseAugmentation]:
    augs = []
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    augs += [Resize(shape=im_hw, keep_ratio=True)]
    return augs


def multi_scale(im_hw: Tuple[int, int]) -> List[BaseAugmentation]:
    augs = []
    augs += [Resize(shape=im_hw, scale_range=(0.8, 1.2), keep_ratio=True)]
    augs += [RandomCrop(shape=im_hw)]
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    return augs


def mosaic_mixup(
    im_hw: Tuple[int, int],
    clip_inside_image: bool = True,
    multiscale_sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[BaseAugmentation]:
    augs = []
    augs += [Mosaic(out_shape=im_hw, clip_inside_image=clip_inside_image)]
    augs += [
        KorniaAugmentationWrapper(
            prob=1.0,
            kornia_type="RandomAffine",
            kwargs={
                "degrees": 10.0,
                "translate": [0.1, 0.1],
                "scale": [0.5, 1.5],
                "shear": [2.0, 2.0],
            },
        )
    ]
    augs += [MixUp(out_shape=im_hw, clip_inside_image=clip_inside_image)]
    augs += [KorniaRandomHorizontalFlip(prob=0.5)]
    if multiscale_sizes is None:
        augs += [Resize(shape=im_hw, keep_ratio=True)]
    else:
        augs += [
            Resize(
                shape=multiscale_sizes, multiscale_mode="list", keep_ratio=True
            )
        ]
    return augs


def add_colorjitter(augs: List[BaseAugmentation]) -> None:
    augs += KorniaColorJitter(
        prob=0.5,
        kwargs={
            "brightness": [0.875, 1.125],
            "contrast": [0.5, 1.5],
            "saturation": [0.5, 1.5],
            "hue": [-0.1, 0.1],
        },
    )
