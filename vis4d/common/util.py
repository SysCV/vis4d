"""Utility functions for common usage."""
import torch

from .imports import is_torch_tf32_available
from .logging import rank_zero_warn


def set_tf32(use_tf32: bool = False) -> None:  # pragma: no cover
    """Set torch TF32.

    Args:
        use_tf32: Whether to use torch TF32.
    """
    if is_torch_tf32_available():  # pragma: no cover
        if use_tf32:
            rank_zero_warn(
                "Torch TF32 is available and turned on by default! "
                + "It might harm the performance due to the precision. "
                + "You can turn it off by setting trainer.use_tf32=False."
            )
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
