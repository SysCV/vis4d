"""Common types for classification models."""

from typing import NamedTuple

import torch


class ClsOut(NamedTuple):
    """Output of the classification results."""

    logits: torch.Tensor  # (N, num_classes)
    probs: torch.Tensor  # (N, num_classes)
