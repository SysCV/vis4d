"""Model weight initialization."""
from torch import nn

def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    """Initialize module with constant value."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)