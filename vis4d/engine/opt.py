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
    ) -> None:
        """Creates an instance of the class."""
        self.learning_rate = learning_rate

        self.model = self.setup_model()
        self.model.to(device)
        if get_world_size() > 1:
            self.model = DDP(self.model, device_ids=[gpu_id])
        self.loss = self.setup_loss()
        self.optimizer = self.setup_optimizer()
        self.lr_scheduler = self.setup_lr_scheduler()
        self.warmup: None | BaseLRWarmup = self.setup_warmup()

    def setup_model(self) -> nn.Module:
        """Set-up model."""
        raise NotImplementedError

    def setup_loss(self) -> nn.Module:
        """Set-up loss function."""
        raise NotImplementedError

    def setup_optimizer(self) -> optim.Optimizer:
        """Set-up optimizer."""
        raise NotImplementedError

    def setup_lr_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Set-up learning rate scheduler."""
        raise NotImplementedError

    def setup_warmup(self) -> None | BaseLRWarmup:
        """Set-up learning rate warm-up."""
        return None

    def warmup_step(self, step: int) -> None:
        """Set learning rate according to warmup."""
        if self.warmup is None:
            return
        if step < 500:
            for g in self.optimizer.param_groups:
                g["lr"] = self.warmup(step, self.learning_rate)
        elif step == 500:
            for g in self.optimizer.param_groups:
                g["lr"] = self.learning_rate
