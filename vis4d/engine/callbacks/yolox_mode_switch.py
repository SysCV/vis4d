"""Callback for switching the mode of YOLOX training."""
from __future__ import annotations

from torch import nn
from torch.utils.data import DataLoader

from vis4d.common import ArgsType
from vis4d.common.logging import rank_zero_info
from vis4d.data.data_pipe import DataPipe
from vis4d.engine.loss_module import LossModule
from vis4d.op.detect.yolox import YOLOXHeadLoss
from vis4d.op.loss.common import l1_loss

from .base import Callback
from .trainer_state import TrainerState


class YOLOXModeSwitchCallback(Callback):
    """Callback for switching the mode of YOLOX training.

    Args:
        switch_epoch (int): Epoch to switch the mode.
    """

    def __init__(
        self, *args: ArgsType, switch_epoch: int, **kwargs: ArgsType
    ) -> None:
        """Init callback."""
        super().__init__(*args, **kwargs)
        self.switch_epoch = switch_epoch
        self.switched = False

    def on_train_epoch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the beginning of a training epoch."""
        found_loss = False
        for loss in loss_module.losses:
            if isinstance(loss["loss"], YOLOXHeadLoss):
                found_loss = True
                yolox_loss = loss["loss"]
                break
        assert found_loss, "YOLOXHeadLoss should be in LossModule."

        if (
            trainer_state["current_epoch"]
        ) >= self.switch_epoch and not self.switched:
            rank_zero_info(
                "Switching YOLOX training mode (turning off strong "
                "augmentations, adding L1 loss)."
            )
            yolox_loss.loss_l1 = l1_loss  # set L1 loss function
            # Set data pipeline to default DataPipe to skip strong augs.
            dataloader = trainer_state["train_dataloader"]
            assert dataloader is not None
            new_dataloader = DataLoader(
                DataPipe(dataloader.dataset.datasets),  # type: ignore
                batch_size=dataloader.batch_size,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                sampler=dataloader.sampler,
                persistent_workers=dataloader.persistent_workers,
                pin_memory=dataloader.pin_memory,
            )
            train_module = trainer_state["train_module"]
            if trainer_state["train_engine"] == "vis4d":
                # Directly modify the train dataloader.
                train_module.train_dataloader = new_dataloader
            self.switched = True

    def on_train_epoch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if (
            trainer_state["current_epoch"]
        ) >= self.switch_epoch - 1 and trainer_state["train_engine"] == "pl":
            # Set data pipeline to default DataPipe to skip strong augs.
            dataloader = trainer_state["train_dataloader"]
            assert dataloader is not None
            new_dataloader = DataLoader(
                DataPipe(dataloader.dataset.datasets),  # type: ignore
                batch_size=dataloader.batch_size,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
                sampler=dataloader.sampler,
                persistent_workers=dataloader.persistent_workers,
                pin_memory=dataloader.pin_memory,
            )
            train_module = trainer_state["train_module"]

            # Override train_dataloader method in PL datamodule.
            # Set reload_dataloaders_every_n_epochs to 1 to use the new
            # dataloader.
            def train_dataloader() -> DataLoader:  # type: ignore
                """Return dataloader for training."""
                return new_dataloader

            train_module.datamodule.train_dataloader = train_dataloader
            train_module.reload_dataloaders_every_n_epochs = 1
