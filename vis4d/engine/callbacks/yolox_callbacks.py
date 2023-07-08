"""YOLOX-specific callbacks."""
from __future__ import annotations

from collections import OrderedDict

from torch import nn
from torch.utils.data import DataLoader

from vis4d.common import ArgsType
from vis4d.common.distributed import all_reduce_dict, get_world_size
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

    def on_train_epoch_end(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if (
            trainer_state["current_epoch"] < self.switch_epoch - 1
            or self.switched
        ):
            # TODO: Make work with resume.
            return

        found_loss = False
        for loss in loss_module.losses:
            if isinstance(loss["loss"], YOLOXHeadLoss):
                found_loss = True
                yolox_loss = loss["loss"]
                break
        assert found_loss, "YOLOXHeadLoss should be in LossModule."
        rank_zero_info(
            "Switching YOLOX training mode starting next training epoch "
            "(turning off strong augmentations, adding L1 loss, switching to "
            "validation every epoch)."
        )
        yolox_loss.loss_l1 = l1_loss  # set L1 loss function
        # Set data pipeline to default DataPipe to skip strong augs.
        # Switch to checking validation every epoch.
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
        train_module.check_val_every_n_epoch = 1
        if trainer_state["train_engine"] == "vis4d":
            # Directly modify the train dataloader.
            train_module.train_dataloader = new_dataloader
        elif trainer_state["train_engine"] == "pl":
            # Override train_dataloader method in PL datamodule.
            # Set reload_dataloaders_every_n_epochs to 1 to use the new
            # dataloader.
            def train_dataloader() -> DataLoader:  # type: ignore
                """Return dataloader for training."""
                return new_dataloader

            train_module.datamodule.train_dataloader = train_dataloader
            train_module.reload_dataloaders_every_n_epochs = self.switch_epoch
        else:
            raise ValueError(
                f"Unsupported training engine {trainer_state['train_engine']}."
            )

        self.switched = True


def get_norm_states(module: nn.Module) -> OrderedDict:
    """Get the state_dict of batch norms in the module."""
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states


class YOLOXSyncNormCallback(Callback):
    """Callback for syncing the norm states of YOLOX training."""

    def on_test_epoch_start(
        self, trainer_state: TrainerState, model: nn.Module
    ) -> None:
        """Hook to run at the beginning of a testing epoch.

        Args:
            trainer_state (TrainerState): Trainer state.
            model (nn.Module): Model that is being trained.
        """
        rank_zero_info("Synced norm states across all processes.")
        world_size = get_world_size()
        if world_size == 1:
            return
        norm_states = get_norm_states(model)
        if len(norm_states) == 0:
            return
        norm_states = all_reduce_dict(norm_states, op="mean")
        model.load_state_dict(norm_states, strict=False)
