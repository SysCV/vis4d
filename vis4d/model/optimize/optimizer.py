"""Vis4D optimizers."""
import re
from typing import Iterator, List, Optional, Tuple

from pydantic import BaseModel
from pytorch_lightning.utilities.distributed import rank_zero_info
from torch import optim
from torch.nn.parameter import Parameter

from vis4d.common.registry import RegistryHolder
from vis4d.struct import DictStrAny


class OptimizerConfig(BaseModel):
    """Config for Vis4D model optimizer."""

    type: str = "SGD"
    lr: float = 1.0e-3
    paramwise_options: Optional[DictStrAny] = None
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


def set_param_wise_optim(
    params: Iterator[Tuple[str, Parameter]], cfg: OptimizerConfig
) -> List[DictStrAny]:
    """Setting param-wise lr and weight decay."""
    if cfg.paramwise_options is None:
        new_params = [{"params": [param]} for _, param in params]
        return new_params

    base_lr = cfg.lr
    base_wd = cfg.kwargs.get("weight_decay", None)

    # get param-wise options
    bias_lr_mult = cfg.paramwise_options.get("bias_lr_mult", 1.0)
    bias_decay_mult = cfg.paramwise_options.get("bias_decay_mult", 1.0)
    norm_decay_mult = cfg.paramwise_options.get("norm_decay_mult", 1.0)
    bboxfc_lr_mult = cfg.paramwise_options.get("bboxfc_lr_mult", 1.0)

    new_params = []
    for name, param in params:
        param_group = {"params": [param]}
        if not param.requires_grad:
            # FP16 training needs to copy gradient/weight between master
            # weight copy and model weight, it is convenient to keep all
            # parameters here to align with model.parameters()
            new_params.append(param_group)
            continue

        bbox_head = (
            name.find("cls") != -1
            or name.find("reg") != -1
            or name.find("dep") != -1
            or name.find("dim") != -1
            or name.find("rot") != -1
            or name.find("2dc") != -1
        )

        # For norm layers, overwrite the weight decay of weight and bias
        # TODO: obtain the norm layer prefixes dynamically  # pylint: disable=line-too-long,fixme
        if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
            if base_wd is not None:
                param_group["weight_decay"] = base_wd * norm_decay_mult
        # For the other layers, overwrite both lr and weight decay of bias
        elif name.endswith(".bias") and name.find("offset") == -1:
            param_group["lr"] = base_lr * bias_lr_mult
            if base_wd is not None:
                param_group["weight_decay"] = base_wd * bias_decay_mult
        # Overwrite bbox head lr
        elif bbox_head:
            param_group["lr"] = base_lr * bboxfc_lr_mult
            rank_zero_info(f"{name} with lr_multi: {bboxfc_lr_mult}")
        new_params.append(param_group)
    return new_params


def build_optimizer(
    params: Iterator[Tuple[str, Parameter]], cfg: OptimizerConfig
) -> BaseOptimizer:
    """Build Optimizer from config."""
    registry = RegistryHolder.get_registry(BaseOptimizer)
    if cfg.type in registry:
        optimizer = registry[cfg.type]  # pragma: no cover
    elif hasattr(optim, cfg.type):
        optimizer = getattr(optim, cfg.type)
    else:
        raise ValueError(f"Optimizer {cfg.type} not known!")
    new_params = set_param_wise_optim(params, cfg)
    module = optimizer(new_params, lr=cfg.lr, **cfg.kwargs)
    return module  # type: ignore
