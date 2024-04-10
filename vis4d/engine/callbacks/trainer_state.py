"""Trainer state for callbacks."""

from __future__ import annotations

from typing import TypedDict

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import NotRequired

from vis4d.common import TrainingModule
from vis4d.data.typing import DictData
from vis4d.engine.optim import LRSchedulerWrapper


class TrainerState(TypedDict):
    """State of the trainer.

    Attributes:
        current_epoch (int): Current epoch.
        num_epochs (int): Total number of the training epochs.
        global_step (int): Global step.
        num_steps (int): Total number of the training steps.
        train_dataloader (DataLoader[DictData] | None): Training dataloader.
        num_train_batches (int | None): Number of training batches.
        test_dataloader (list[DataLoader[DictData]] | None): List of test
            dataloaders.
        num_test_batches (list[int] | None): List of number of test batches.
        optimizers (NotRequired[list[Optimizer]]): List of optimizers.
        metrics (NotRequired[dict[str, float]]): Metrics for the logging.
    """

    current_epoch: int
    num_epochs: int
    global_step: int
    num_steps: int
    train_dataloader: DataLoader[DictData] | None
    num_train_batches: int | None
    test_dataloader: list[DataLoader[DictData]] | None
    num_test_batches: list[int] | None
    optimizers: NotRequired[list[Optimizer]]
    lr_schedulers: NotRequired[list[LRSchedulerWrapper]]
    metrics: NotRequired[dict[str, float]]
    train_module: NotRequired[TrainingModule]
    train_engine: NotRequired[str]
