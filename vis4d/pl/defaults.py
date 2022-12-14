# pylint: disable=consider-using-alias,consider-alternative-union-syntax
"""Common optimizers."""
from typing import List, Optional

from vis4d.common import DictStrAny


def sgd(  # pylint: disable=invalid-name
    lr: float, momentum: float = 0.9, weight_decay: float = 0.0001
) -> DictStrAny:
    """Standard SGD optimizer cfg with given lr."""
    optimizer_cfg = {
        "class_path": "torch.optim.SGD",
        "init_args": {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
    }
    return optimizer_cfg


def adam(  # pylint: disable=invalid-name
    lr: float, amsgrad: bool = False, weight_decay: float = 0.0001
) -> DictStrAny:
    """Standard Adam optimizer cfg with given lr."""
    optimizer_cfg = {
        "class_path": "torch.optim.Adam",
        "init_args": {
            "lr": lr,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        },
    }
    return optimizer_cfg


def adamW(  # pylint: disable=invalid-name
    lr: float, weight_decay: float = 0.0001, epsilon: float = 1e-8
) -> DictStrAny:
    """Standard AdamW optimizer cfg with given lr."""
    optimizer_cfg = {
        "class_path": "torch.optim.AdamW",
        "init_args": {
            "lr": lr,
            "weight_decay": weight_decay,
            "eps": epsilon,
        },
    }
    return optimizer_cfg


def step_schedule(
    max_steps: int = 12,
    milestones: Optional[List[int]] = None,
    gamma: float = 0.1,
    mode: str = "epoch",
) -> DictStrAny:
    """Create standard step schedule cfg according to max epochs."""
    if milestones is None:
        milestones = [int(max_steps * 2 / 3), int(max_steps * 11 / 12)]
    lr_scheduler_cfg = {
        "class_path": "torch.optim.lr_scheduler.MultiStepLR",
        "init_args": {"milestones": milestones, "gamma": gamma},
        "mode": mode,
    }
    return lr_scheduler_cfg


def poly_schedule(
    max_steps: int = 40000, power: float = 0.9, min_lr: float = 0.0001
) -> DictStrAny:
    """Create poly schedule cfg."""
    lr_scheduler_cfg = {
        "class_path": "vis4d.op.optimize.scheduler.PolyLRScheduler",
        "init_args": {
            "max_steps": max_steps,
            "power": power,
            "min_lr": min_lr,
        },
        "mode": "step",
    }
    return lr_scheduler_cfg
