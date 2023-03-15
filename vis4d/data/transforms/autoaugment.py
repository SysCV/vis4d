"""A wrap for timm transforms."""
from typing import Union

import numpy as np
import torch
from PIL import Image

from vis4d.common.imports import TIMM_AVAILABLE
from vis4d.data.const import CommonKeys as K

from .base import Transform

if TIMM_AVAILABLE:
    from timm.data.auto_augment import (
        _RAND_INCREASING_TRANSFORMS,
        _RAND_TRANSFORMS,
        AugMixAugment,
        AutoAugment,
        RandAugment,
        augmix_ops,
        auto_augment_policy,
        rand_augment_ops,
    )

AugOp = Union[AutoAugment, RandAugment, AugMixAugment]


def _apply_aug(images: torch.Tensor, aug_op: AugOp) -> torch.Tensor:
    """Apply augmentation to a batch of images."""
    pil_imgs = [
        Image.fromarray(image)
        for image in images.permute(0, 2, 3, 1).cpu().numpy()
    ]
    for i, img in enumerate(pil_imgs):
        pil_imgs[i] = aug_op(img)
    return torch.stack(
        [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in pil_imgs]
    )


@Transform(
    in_keys=(K.images,),
    out_keys=(K.images,),
)
def autoaug(policy: str, magnitude_std: float = 0.5):
    """Apply timm's autoaugment to a image tensor.

    Args:
        policy (str): Policy name for autoaugment. Options are "original",
            "originalr", "v0", "v0r".
        magnitude_std (float, optional): Standard deviation of the magnitude
            for random autoaugment. Defaults to 0.5.

    Returns:
        Callable: A function that takes a tensor of shape [N, C, H, W] and
            returns a tensor of the same shape.

    Example:
        Original autoaugment policy (`https://arxiv.org/abs/1805.09501`)
        >>> autoaugment("original", magnitude_std=0.5)
    """
    assert TIMM_AVAILABLE, "timm is not installed."
    assert policy in {
        "original",
        "originalr",
        "v0",
        "v0r",
    }, "Policy must be one of 'original', 'originalr', 'v0', 'v0r'."

    def _autoaugment(images: torch.Tensor) -> torch.Tensor:
        hparams = {"magnitude_std": magnitude_std}
        aa_policy = auto_augment_policy(policy, hparams=hparams)
        aug_op = AutoAugment(aa_policy)
        pil_imgs = [
            Image.fromarray(image)
            for image in images.permute(0, 2, 3, 1).cpu().numpy()
        ]
        for i, img in enumerate(pil_imgs):
            pil_imgs[i] = aug_op(img)
        return torch.stack(
            [
                torch.from_numpy(np.array(img)).permute(2, 0, 1)
                for img in pil_imgs
            ]
        )

    return _autoaugment


@Transform(
    in_keys=(K.images,),
    out_keys=(K.images,),
)
def randaug(
    magnitude: int,
    num_layers: int = 2,
    use_increasing: bool = False,
    magnitude_std: float = 0.5,
):
    """Apply timm's rand augment to a image tensor.

    Args:
        magnitude (int): Level of magnitude for augments, ranging from 1 to 9.
        num_layers (int, optional): Number of layers for rand augment. Defaults
            to 2.
        use_increasing (bool, optional): Whether to use increasing setting for
            transforms. Defaults to False.
        magnitude_std (float, optional): Standard deviation of the magnitude
            for random autoaugment. Defaults to 0.5.

    Returns:
        Callable: A function that takes a tensor of shape [N, C, H, W] and
            returns a tensor of the same shape.

    Example:
        Rand augment with magnitude 9. (`https://arxiv.org/abs/1909.13719`)
        >>> rand_augment(magnitude=9)
    """
    assert TIMM_AVAILABLE, "timm is not installed."

    def _rand_augment(images: torch.Tensor) -> torch.Tensor:
        hparams = {"magnitude_std": magnitude_std}
        if use_increasing:
            transforms = _RAND_INCREASING_TRANSFORMS
        else:
            transforms = _RAND_TRANSFORMS
        ra_ops = rand_augment_ops(
            magnitude=magnitude, hparams=hparams, transforms=transforms
        )
        aug_op = RandAugment(ra_ops, num_layers)
        return _apply_aug(images, aug_op)

    return _rand_augment


@Transform(
    in_keys=(K.images,),
    out_keys=(K.images,),
)
def augmix(
    magnitude: int,
    width: int = 3,
    alpha: float = 1.0,
    depth: int = -1,
    blended: bool = True,
    magnitude_std: float = 0.5,
):
    """Apply timm's augmix to a image tensor.

    Args:
        magnitude (int): Level of magnitude for augments, ranging from 1 to 9.
        width (int, optional): Width of the augmentation chain. Defaults to 3.
        alpha (float, optional): Alpha for beta distribution. Defaults to 1.0.
        depth (int, optional): Depth of the augmentation chain. Defaults to -1.
        blended (bool, optional): Whether to blend the original image with the
            augmented image. Defaults to True.
        magnitude_std (float, optional): Standard deviation of the magnitude
            for random autoaugment. Defaults to 0.5.

    Returns:
        Callable: A function that takes a tensor of shape [N, C, H, W] and
            returns a tensor of the same shape.

    Example:
        Augmix with magnitude 9. (`https://arxiv.org/abs/1912.02781`)
        >>> augmix(magnitude=9)
    """
    assert TIMM_AVAILABLE, "timm is not installed."

    def _augmix(images: torch.Tensor) -> torch.Tensor:
        hparams = {"magnitude_std": magnitude_std}
        am_ops = augmix_ops(magnitude=magnitude, hparams=hparams)
        aug_op = AugMixAugment(
            am_ops, alpha=alpha, width=width, depth=depth, blended=blended
        )
        return _apply_aug(images, aug_op)

    return _augmix
