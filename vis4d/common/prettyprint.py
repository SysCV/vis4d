"""Utilities for pretty printing."""
import numpy as np
import torch


class PrettyRepMixin:
    """Creates a pretty string representation of a class with parameters."""

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        attr_str = ""
        for k, v in vars(self).items():
            if k != "type" and not k.startswith("_"):
                attr_str += f"{k}={str(v)}, "
        attr_str = attr_str.rstrip(", ")
        return f"{self.__class__.__name__}({attr_str})"


def describe_shape(obj):
    """Recursively output the shape of tensors in its structure."""

    log_str = ""
    if isinstance(obj, dict):
        log_str += "{"
        for k in obj.keys():
            log_str += f"{k}: {describe_shape(obj[k])}, "
        log_str += "}"
    elif isinstance(obj, list):
        log_str += "["
        for v in obj:
            log_str += describe_shape(v) + ", "
        log_str += "]"
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        log_str += f"shape[{', '.join([str(s) for s in obj.shape])}]"
    elif isinstance(obj, float):
        log_str += f"{obj:.4f}"
    else:
        log_str += str(obj)
    return log_str
