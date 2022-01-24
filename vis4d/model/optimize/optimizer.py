"""Vis4D optimizers."""
from typing import Iterator, Tuple

from pydantic import BaseModel
from torch import optim
from torch.nn.parameter import Parameter

from vis4d.common.registry import RegistryHolder
from vis4d.struct import DictStrAny


class OptimizerConfig(BaseModel):
    """Config for Vis4D model optimizer."""

    type: str = "SGD"
    lr: float = 1.0e-3
    kwargs: DictStrAny = {
        "momentum": 0.9,
        "weight_decay": 0.0001,
    }


class BaseOptimizer(optim.Optimizer, metaclass=RegistryHolder):  # type: ignore
    """Dummy Optimizer class supporting Vis4D registry."""

    def step(self, closure):  # type: ignore
        """Performs a single parameter update.

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError


def build_optimizer(
    params: Iterator[Parameter], cfg: OptimizerConfig
) -> BaseOptimizer:
    """Build Optimizer from config."""
    registry = RegistryHolder.get_registry(BaseOptimizer)
    if cfg.type in registry:
        optimizer = registry[cfg.type]  # pragma: no cover
    elif hasattr(optim, cfg.type):
        optimizer = getattr(optim, cfg.type)
    else:
        raise ValueError(f"Optimizer {cfg.type} not known!")
    module = optimizer(params, lr=cfg.lr, **cfg.kwargs)
    return module  # type: ignore
