"""Mixup data augmentation."""

import numpy as np

from vis4d.common.typing import NDArrayF32, NDArrayI32
from vis4d.data.const import CommonKeys as K

from .base import BatchTransform


@BatchTransform(
    in_keys=(K.images, K.categories), out_keys=(K.images, K.smooth_categories)
)
class Mixup:
    """Mixup data augmentation."""

    def __init__(
        self,
        probability: float = 0.2,
        alpha: float = 1.0,
        label_smoothing: float = 0.1,
        num_classes: int = 1000,
    ):
        """Creates an instance of Mixup.

        Args:
            probability (float, optional): Probability of applying mixup.
            alpha (float, optional): Alpha value for beta distribution.
                Defaults to 1.0.
            label_smoothing (float, optional): Label smoothing value for the
                target. Defaults to 0.1.
            num_classes (int, optional): Number of classes. Defaults to 1000.
        """
        self.probability = probability
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _label_smoothing(
        self,
        cat_1: NDArrayI32,
        cat_2: NDArrayI32,
        lambda_: float,
    ) -> NDArrayF32:
        """Apply label smoothing to two category labels."""
        lam = np.array(lambda_, dtype=np.float32)
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
        categories_1[cat_1] = on_value
        categories_2[cat_2] = on_value
        smoothed = categories_1 * lam + categories_2 * (1 - lam)
        return smoothed.astype(np.float32)

    def _get_lambda(self, batch_size: int) -> NDArrayF32:
        """Get lambda values for mixup."""
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        return np.maximum(lam, 1 - lam)

    def __call__(
        self, images: list[NDArrayF32], categories: list[NDArrayI32]
    ) -> tuple[list[NDArrayF32], list[NDArrayF32]]:
        """Execute mixup op."""
        batch_size = len(images)
        assert batch_size % 2 == 0, "Batch size must be even for mixup!"

        if np.random.rand() > self.probability:
            _eye = np.eye(self.num_classes, dtype=np.float32)
            smooth_categories = [_eye[cat] for cat in categories]
            return images, smooth_categories

        smooth_categories = [np.empty(0, dtype=np.float32)] * batch_size
        lam = self._get_lambda(batch_size // 2)
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            images[i] = images[i] * lam[i] + images[j] * (1 - lam[i])
            images[j] = images[j] * lam[i] + images[i] * (1 - lam[i])
            smooth_cat_i = self._label_smoothing(
                categories[i], categories[j], lam[i]
            )
            smooth_cat_j = self._label_smoothing(
                categories[j], categories[i], lam[i]
            )
            smooth_categories[i] = smooth_cat_i
            smooth_categories[j] = smooth_cat_j
        return images, smooth_categories
