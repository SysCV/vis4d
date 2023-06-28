"""Optimizer."""
from __future__ import annotations

from typing import TypedDict

from ml_collections import ConfigDict
from torch import nn
from torch.optim.optimizer import Optimizer
from typing_extensions import NotRequired

from vis4d.common.logging import rank_zero_info
from vis4d.config import instantiate_classes

from .scheduler import LRSchedulerWrapper


class ParamGroupsCfg(TypedDict):
    """Parameter groups config.

    Attributes:
        custom_keys (list[str]): List of custom keys.
        lr_mult (NotRequired[float]): Learning rate multiplier.
        decay_mult (NotRequired[float]): Weight Decay multiplier.
    """

    custom_keys: list[str]
    lr_mult: NotRequired[float]
    decay_mult: NotRequired[float]


class ParamGroup(TypedDict):
    """Parameter dictionary.

    Attributes:
        params (list[nn.Parameter]): List of parameters.
        lr (NotRequired[float]): Learning rate.
        weight_decay (NotRequired[float]): Weight decay.
    """

    params: list[nn.Parameter]
    lr: NotRequired[float]
    weight_decay: NotRequired[float]


# TODO: Add true support for multiple optimizers. This will need to
# modify config to specify which optimizer to use for which module.
def set_up_optimizers(
    optimizers_cfg: list[ConfigDict], models: list[nn.Module]
) -> tuple[list[Optimizer], list[LRSchedulerWrapper]]:
    """Set up optimizers."""
    optimizers = []
    lr_schedulers = []
    for optim_cfg, model in zip(optimizers_cfg, models):
        optimizer = configure_optimizer(optim_cfg, model)
        optimizers.append(optimizer)

        if optim_cfg.lr_schedulers is not None:
            lr_schedulers.append(
                LRSchedulerWrapper(optim_cfg.lr_schedulers, optimizer)
            )

    return optimizers, lr_schedulers


def configure_optimizer(optim_cfg: ConfigDict, model: nn.Module) -> Optimizer:
    """Configure optimizer with parameter groups."""
    param_groups_cfg = optim_cfg.get("param_groups", None)

    if param_groups_cfg is not None:
        params = []
        base_lr = optim_cfg.optimizer["init_args"].lr
        weight_decay = optim_cfg.optimizer["init_args"].get(
            "weight_decay", None
        )
        for group in param_groups_cfg:
            lr_mult = group.get("lr_mult", 1.0)
            decay_mult = group.get("decay_mult", 1.0)

            param_group: ParamGroup = {
                "params": [],
                "lr": base_lr * lr_mult,
            }

            if weight_decay is not None:
                param_group["weight_decay"] = weight_decay * decay_mult

            params.append(param_group)

        # Create a param group for the rest of the parameters
        param_group = {"params": [], "lr": base_lr}
        if weight_decay is not None:
            param_group["weight_decay"] = weight_decay
        params.append(param_group)

        # Add the parameters to the param groups
        add_params(params, model, param_groups_cfg)

        return instantiate_classes(optim_cfg.optimizer, params=params)

    return instantiate_classes(optim_cfg.optimizer, params=model.parameters())


def add_params(
    params: list[ParamGroup],
    module: nn.Module,
    param_groups_cfg: list[ParamGroupsCfg],
    prefix: str = "",
) -> None:
    """Add all parameters of module to the params list.

    The parameters of the given module will be added to the list of param
    groups, with specific rules defined by paramwise_cfg.

    Args:
        params (list[DictStrAny]): A list of param groups, it will be modified
            in place.
        module (nn.Module): The module to be added.
        param_groups_cfg (dict[str, list[str] | float]): The configuration
            of the param groups.
        prefix (str): The prefix of the module. Default: ''.
    """
    for name, param in module.named_parameters(recurse=False):
        if not param.requires_grad:
            params[-1]["params"].append(param)
            continue

        # if the parameter match one of the custom keys, ignore other rules
        is_custom = False
        msg = f"{prefix}.{name}"
        for i, group in enumerate(param_groups_cfg):
            for key in group["custom_keys"]:
                if key in f"{prefix}.{name}":
                    if group.get("lr_mult", None) is not None:
                        msg += f" with lr_mult: {group['lr_mult']}"
                    if group.get("decay_mult", None) is not None:
                        msg += f" with decay_mult: {group['decay_mult']}"
                    params[i]["params"].append(param)
                    is_custom = True
                    break
            if is_custom:
                break

        if is_custom:
            rank_zero_info(msg)
        else:
            params[-1]["params"].append(param)

    for child_name, child_mod in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        add_params(params, child_mod, param_groups_cfg, prefix=child_prefix)
