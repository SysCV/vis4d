"""A wrap for timm transforms."""
import numpy as np
import torch
from PIL import Image

from vis4d.common.imports import TIMM_AVAILABLE
from vis4d.data.const import CommonKeys as K

from .base import Transform

if TIMM_AVAILABLE:
    from timm.data.auto_augment import (
        augment_and_mix_transform,
        auto_augment_transform,
        rand_augment_transform,
    )


@Transform(
    in_keys=(K.images,),
    out_keys=(K.images,),
)
def autoaugment(policy: str, params: dict[str, float]):
    """Apply timm's autoaugment to a image tensor.

    Args:
        policy (str): Policy name for autoaugment. String defining
            configuration of auto augmentation. Consists of multiple sections
            separated by dashes ('-'). The first section defines the
            AutoAugment policy o use. The remaining sections, if present,
            define the augmentation parameters.
        params (dict[str, float], optional): Parameters for the policy.

    Returns:
        Callable: A function that takes a tensor of shape [N, C, H, W] and
            returns a tensor of the same shape.

    Example:
        Original autoaugment policy (`https://arxiv.org/abs/1805.09501`)
        >>> autoaugment("original")

        Autoaugment policy used in EfficientNet implementation
        >>> autoaugment("v0")

        Random autoaugment policy (`https://arxiv.org/abs/1909.13719`)
        >>> autoaugment("rand-m9-mstd0.5")

        AugMix policy
        >>> autoaugment("augmix-m1-mstd0.5")
    """

    def _autoaugment(images: torch.Tensor) -> torch.Tensor:
        if policy.startswith("rand"):
            timm_transform = rand_augment_transform
        elif policy.startswith("augmix"):
            timm_transform = augment_and_mix_transform
        else:
            timm_transform = auto_augment_transform
        aug_op = timm_transform(policy, params)
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
