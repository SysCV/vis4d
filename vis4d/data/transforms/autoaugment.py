"""A wrap for timm transforms."""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

from vis4d.common.imports import TIMM_AVAILABLE
from vis4d.common.typing import NDArrayUI8
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
else:
    raise ImportError("timm is not installed.")

AugOp = Union[AutoAugment, RandAugment, AugMixAugment]


def _apply_aug(images: NDArrayUI8, aug_op: AugOp) -> NDArrayUI8:
    """Apply augmentation to a batch of images with shape [N, H, W, C]."""
    assert images.shape[-1] == 3, "Images must be in RGB format."
    imgs: list[Image.Image] = []
    for img in images:
        # convert to uint8 if necessary
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        imgs.append(aug_op(Image.fromarray(img)))
    return np.stack([np.array(img).astype(np.float32) for img in imgs])


@Transform(K.images, K.images)
class _AutoAug:
    """Apply Timm's AutoAugment to a image array."""

    def __init__(self) -> None:
        self.aug_op: AugOp | None = None

    def _create(self, policy: str, hparams: dict[str, float]) -> AugOp:
        """Create augmentation op."""
        aa_policy = auto_augment_policy(policy, hparams=hparams)
        return AutoAugment(aa_policy)

    def __call__(self, images: list[NDArrayUI8]) -> list[NDArrayUI8]:
        """Execute the transform."""
        assert self.aug_op is not None, "Augmentation op is not created."
        for i, img in enumerate(images):
            images[i] = _apply_aug(img, self.aug_op)
        return images


class AutoAugV0(_AutoAug):
    """Apply Timm's AutoAugment (policy=v0) to a image array."""

    def __init__(self, magnitude_std: float = 0.5):
        """Create an instance of AutoAug.

        Args:
            magnitude_std (float, optional): Standard deviation of the
                magnitude for random autoaugment. Defaults to 0.5.
        """
        super().__init__()
        self.aug_op = self._create("v0", {"magnitude_std": magnitude_std})


class AutoAugOriginal(_AutoAug):
    """Apply Timm's AutoAugment (policy=original) to a image array."""

    def __init__(self, magnitude_std: float = 0.5):
        """Create an instance of AutoAug.

        Args:
            magnitude_std (float, optional): Standard deviation of the
                magnitude for random autoaugment. Defaults to 0.5.
        """
        super().__init__()
        self.aug_op = self._create(
            "original", {"magnitude_std": magnitude_std}
        )


@Transform(K.images, K.images)
class RandAug:
    """Apply Timm's RandomAugment to a image tensor."""

    def __init__(
        self,
        magnitude: int = 10,
        num_layers: int = 2,
        use_increasing: bool = False,
        magnitude_std: float = 0.5,
    ):
        """Create an instance of RandAug.

        Args:
            magnitude (int): Level of magnitude for augments, ranging from 1 to
                9.
            num_layers (int, optional): Number of layers for rand augment.
                Defaults to 2.
            use_increasing (bool, optional): Whether to use increasing setting
                for transforms. Defaults to False.
            magnitude_std (float, optional): Standard deviation of the
                magnitude for random autoaugment. Defaults to 0.5.

        Returns:
            Callable: A function that takes a tensor of shape [N, C, H, W] and
                returns a tensor of the same shape.

        Example:
            Rand augment with magnitude 9. (`https://arxiv.org/abs/1909.13719`)
            >>> rand_augment(magnitude=9)
        """
        super().__init__()
        assert TIMM_AVAILABLE, "timm is not installed."
        self.magnitude = magnitude
        self.num_layers = num_layers
        self.use_increasing = use_increasing
        self.magnitude_std = magnitude_std
        hparams = {"magnitude_std": self.magnitude_std}

        if self.use_increasing:
            transforms = _RAND_INCREASING_TRANSFORMS
        else:
            transforms = _RAND_TRANSFORMS
        ra_ops = rand_augment_ops(
            magnitude=self.magnitude, hparams=hparams, transforms=transforms
        )
        self.aug_op = RandAugment(ra_ops, self.num_layers)

    def __call__(self, images: list[NDArrayUI8]) -> list[NDArrayUI8]:
        """Execute the transform."""
        for i, img in enumerate(images):
            images[i] = _apply_aug(img, self.aug_op)
        return images


@Transform(K.images, K.images)
class AugMix:
    """Apply Timm's AugMix to a image tensor."""

    def __init__(
        self,
        magnitude: int = 10,
        width: int = 3,
        alpha: float = 1.0,
        depth: int = -1,
        blended: bool = True,
        magnitude_std: float = 0.5,
    ):
        """Create an instance of AugMix.

        Args:
            magnitude (int): Level of magnitude, ranging from 1 to 9.
            width (int, optional): Width of the augmentation chain. Defaults to
                3.
            alpha (float, optional): Alpha for beta distribution. Defaults to
                1.0.
            depth (int, optional): Depth of the augmentation chain. Defaults to
                -1.
            blended (bool, optional): Whether to blend the original image with
                the augmented image. Defaults to True.
            magnitude_std (float, optional): Standard deviation of the
                magnitude for random autoaugment. Defaults to 0.5.

        Returns:
            Callable: A function that takes a tensor of shape [N, C, H, W] and
                returns a tensor of the same shape.

        Example:
            Augmix with magnitude 9. (`https://arxiv.org/abs/1912.02781`)
            >>> augmix(magnitude=9)
        """
        super().__init__()
        assert TIMM_AVAILABLE, "timm is not installed."
        self.magnitude = magnitude
        self.width = width
        self.alpha = alpha
        self.depth = depth
        self.blended = blended
        self.magnitude_std = magnitude_std
        hparams = {"magnitude_std": self.magnitude_std}

        am_ops = augmix_ops(magnitude=self.magnitude, hparams=hparams)
        self.aug_op = AugMixAugment(
            am_ops,
            alpha=self.alpha,
            width=self.width,
            depth=self.depth,
            blended=self.blended,
        )

    def __call__(self, images: list[NDArrayUI8]) -> list[NDArrayUI8]:
        """Execute the transform."""
        for i, img in enumerate(images):
            images[i] = _apply_aug(img, self.aug_op)
        return images
