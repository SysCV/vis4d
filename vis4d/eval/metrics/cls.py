"""Classification metrics."""

from __future__ import annotations

import numpy as np

from vis4d.common.array import array_to_numpy
from vis4d.common.typing import ArrayLike, ArrayLikeInt


def accuracy(
    prediction: ArrayLike, target: ArrayLikeInt, top_k: int = 1
) -> float:
    """Calculate the accuracy of the prediction.

    Args:
        prediction (ArrayLike): Probabilities (or logits) of shape (N, C) or
            (C, ).
        target (ArrayLikeInt): Target of shape (N, ) or (1, ).
        top_k (int, optional): Top k accuracy. Defaults to 1.

    Returns:
        float: Accuracy of the prediction, in range [0, 1].
    """
    prediction = array_to_numpy(prediction, n_dims=2, dtype=np.float32)
    target = array_to_numpy(target, n_dims=1, dtype=np.int64)
    assert prediction.shape[0] == target.shape[0], "Batch size mismatch."
    top_k = min(top_k, prediction.shape[1])
    top_k_idx = np.argsort(prediction, axis=1)[:, -top_k:]
    correct = np.any(top_k_idx == target[:, None], axis=1)
    return float(np.mean(correct))
