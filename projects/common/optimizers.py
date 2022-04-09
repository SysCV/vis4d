"""Common optimizers."""
from vis4d.struct import DictStrAny


def sgd(
    lr: float, momentum: float = 0.9, weight_decay: float = 0.0001
) -> DictStrAny:
    """Standard SGD optimizer cfg with given lr."""
    lr_scheduler_cfg = {
        "class_path": "torch.optim.SGD",
        "init_args": {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
    }
    return lr_scheduler_cfg


def step_schedule(max_epochs: int = 12) -> DictStrAny:
    """Create standard step schedule cfg according to max epochs."""
    lr_scheduler_cfg = {
        "class_path": "torch.optim.lr_scheduler.MultiStepLR",
        "init_args": {
            "milestones": [int(max_epochs * 2 / 3), int(max_epochs * 11 / 12)]
        },
    }
    return lr_scheduler_cfg
