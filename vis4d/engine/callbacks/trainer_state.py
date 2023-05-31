"""Trainer state for callbacks."""
from __future__ import annotations

from typing import TypedDict

from torch.utils.data import DataLoader
from typing_extensions import NotRequired

from vis4d.data import DictData


class TrainerState(TypedDict):
    """State of the trainer.

    Attributes:
        current_epoch (int): Current epoch.
        num_epochs (int): Total number of the training epochs.
        global_step (int): Global step.
        train_dataloader (DataLoader[DictData] | None): Training dataloader.
        num_train_batches (int | None): Number of training batches.
        test_dataloader (list[DataLoader[DictData]] | None): List of test
            dataloaders.
        num_test_batches (list[int] | None): List of number of test batches.
        metrics (NotRequired[dict[str, float]]): Metrics for the logging.
    """

    current_epoch: int
    num_epochs: int
    global_step: int
    train_dataloader: DataLoader[DictData] | None
    num_train_batches: int | None
    test_dataloader: list[DataLoader[DictData]] | None
    num_test_batches: list[int] | None
    metrics: NotRequired[dict[str, float]]
