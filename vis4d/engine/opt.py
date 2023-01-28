"""Vis4D optimizer."""
from __future__ import annotations

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from vis4d.common.distributed import get_world_size
from vis4d.optim.warmup import BaseLRWarmup


class Optimizer:
    """Vis4D Optimizer."""

    def __init__(
        self,
        learning_rate: float,
        device: None | torch.device,
        gpu_id: None | int,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        lr_warmup: None | BaseLRWarmup,
        lr_scheduler: None | optim.lr_scheduler._LRScheduler,
    ) -> None:
        """Creates an instance of the class."""
        self.learning_rate = learning_rate

        self.model = model.to(device)

        if get_world_size() > 1:
            assert gpu_id is not None, "GPU ID must be specified for DDP."
            # This is ok. Not sure why pylint complains.
            self.model = DDP(self.model, device_ids=[gpu_id])
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warmup = lr_warmup

    def warmup_step(self, step: int) -> None:
        """Set learning rate according to warmup."""
        if self.warmup is None:
            return
        if step < 500:
            for g in self.optimizer.param_groups:
                g["lr"] = self.warmup(  # pylint: disable=not-callable
                    step, self.learning_rate
                )
        elif step == 500:
            for g in self.optimizer.param_groups:
                g["lr"] = self.learning_rate
