"""Utility functions for common usage."""
import random
from difflib import get_close_matches

import numpy as np
import torch

from .imports import is_torch_tf32_available
from .logging import rank_zero_warn


def create_did_you_mean_msg(keys: list[str], query: str) -> str:
    """Create a did you mean message.

    Args:
        keys (list[str]): List of available keys.
        query (str): Query.

    Returns:
        str: Did you mean message.

    Examples:
        >>> keys = ["foo", "bar", "baz"]
        >>> query = "fo"
        >>> print(create_did_you_mean_msg(keys, query))
        Did you mean:
            foo
    """
    msg = ""
    if len(keys) > 0:
        msg = "Did you mean:\n\t"
        msg += "\n\t".join(get_close_matches(query, keys, cutoff=0.75))
    return msg


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


def init_random_seed() -> int:
    """Initialize random seed for the experiment."""
    return np.random.randint(2**31)


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
