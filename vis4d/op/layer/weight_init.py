"""Model weight initialization."""
from torch import nn


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    """Initialize module with constant value."""
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    """Initialize module with Xavier initialization."""
    assert distribution in {"uniform", "normal"}
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    negative_slope: float = 0.0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0.0,
    distribution: str = "normal",
) -> None:
    """Initialize module with Kaiming initialization."""
    assert distribution in {"uniform", "normal"}
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight,
                a=negative_slope,
                mode=mode,
                nonlinearity=nonlinearity,
            )
        else:
            nn.init.kaiming_normal_(
                module.weight,
                a=negative_slope,
                mode=mode,
                nonlinearity=nonlinearity,
            )
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)


def normal_init(
    module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0
) -> None:
    """Initialize module with normal distribution."""
    if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
        nn.init.constant_(module.bias, bias)
