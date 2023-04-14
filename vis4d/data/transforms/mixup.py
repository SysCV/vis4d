"""Mixup data augmentation."""
from typing import TypedDict

import numpy as np

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K

from .base import BatchTransform


class MixupParam(TypedDict):
    """Typed dict for mixup ratio.

    The ratio is a float in the range [0, 1] that is used to mixup a pair of
    items in a batch. Usually, the pairs are selected as follows:
        (0, bs - 1), (1, bs - 2), ..., (bs // 2, bs // 2)
    where bs is the batch size. The batch size must be even for mixup to work.
    """

    ratio: NDArrayF32  # shape (batch_size,)


@BatchTransform(in_keys=(K.images,), out_keys=("transforms.mixup",))
class GenMixupParameters:
    """Generate the parameters for a mixup operation."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Creates an instance of GenMixupParameters.

        Args:
            alpha (float, optional): Parameter for beta distribution used for
                sampling the mixup ratio (i.e., lambda). Defaults to 1.0.
        """
        self.alpha = alpha

    def __call__(self, images: list[NDArrayF32]) -> list[MixupParam]:
        """Generate the mixup parameters."""
        batch_size = len(images)
        ratio = np.random.beta(self.alpha, self.alpha, batch_size).astype(
            np.float32
        )
        ratio = np.maximum(ratio, 1 - ratio)
        return [MixupParam(ratio=ratio)] * batch_size


@BatchTransform(
    in_keys=(K.images, "transforms.mixup.ratio"),
    out_keys=(K.images,),
)
class MixupImages:
    """Mixup a batch of images."""

    def __call__(
        self,
        images: list[NDArrayF32],
        ratios: list[NDArrayF32],
    ) -> list[NDArrayF32]:
        """Execute image mixup operation."""
        batch_size = len(images)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        ratio = ratios[0]
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            images[i] = images[i] * ratio[i] + images[j] * (1 - ratio[i])
            images[j] = images[j] * ratio[i] + images[i] * (1 - ratio[i])
        return images


@BatchTransform(
    in_keys=(K.categories, "transforms.mixup.ratio"),
    out_keys=(K.categories,),
)
class MixupCategories:
    """Mixup a batch of categories."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.1) -> None:
        """Creates an instance of MixupCategories.

        Args:
            num_classes (int): Number of classes.
            label_smoothing (float, optional): Label smoothing parameter for
                the mixup of categories. Defaults to 0.1.
        """
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def _label_smoothing(
        self,
        cat_1: NDArrayF32,
        cat_2: NDArrayF32,
        ratio: float,
    ) -> NDArrayF32:
        """Apply label smoothing to two category labels."""
        lam = np.array(ratio, dtype=np.float32)
        off_value = np.array(
            self.label_smoothing / self.num_classes, dtype=np.float32
        )
        on_value = np.array(
            1 - self.label_smoothing + off_value, dtype=np.float32
        )
        categories_1: NDArrayF32 = (
            np.zeros((self.num_classes,), dtype=np.float32) + off_value
        )
        categories_2: NDArrayF32 = (
            np.zeros((self.num_classes,), dtype=np.float32) + off_value
        )
        categories_1 = cat_1 * on_value
        categories_2 = cat_2 * on_value
        mixed = categories_1 * lam + categories_2 * (1 - lam)
        return mixed.astype(np.float32)

    def __call__(
        self,
        categories: list[NDArrayF32],
        ratios: list[NDArrayF32],
    ) -> list[NDArrayF32]:
        """Execute the categories mixup operation."""
        batch_size = len(categories)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        ratio = ratios[0]
        smooth_categories = [np.empty(0, dtype=np.float32)] * batch_size
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            smooth_categories[i] = self._label_smoothing(
                categories[i], categories[j], ratio[i]
            )
            smooth_categories[j] = self._label_smoothing(
                categories[j], categories[i], ratio[i]
            )
        return smooth_categories
