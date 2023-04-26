"""Trainer state for callbacks."""
from typing import TypedDict
from typing_extensions import NotRequired

from torch import Tensor
from torch.utils.data import DataLoader

from vis4d.data import DictData
from vis4d.engine.connectors import DataConnector


class TrainerState(TypedDict):
    """State of the trainer."""

    current_epoch: int
    num_epochs: int
    global_steps: int
    data_connector: DataConnector
    train_dataloader: DataLoader[DictData] | None
    num_train_batches: int | None
    test_dataloader: list[DataLoader[DictData]] | None
    num_test_batches: list[int] | None
    metrics: NotRequired[dict[str, Tensor]]
