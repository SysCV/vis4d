"""YOLOX-specific callbacks."""

from __future__ import annotations

import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _NormBase
from torch.utils.data import DataLoader

from vis4d.common import ArgsType, DictStrAny
from vis4d.common.distributed import (
    all_reduce_dict,
    broadcast,
    get_rank,
    get_world_size,
    synchronize,
)
from vis4d.common.logging import rank_zero_info, rank_zero_warn
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.typing import DictDataOrList
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
        rank_zero_info(
            "Switching YOLOX training mode starting next training epoch "
            "(turning off strong augmentations, adding L1 loss, switching to "
            "validation every epoch)."
        )
        if found_loss:
            yolox_loss.loss_l1 = l1_loss  # set L1 loss function
        else:
            rank_zero_warn("YOLOXHeadLoss should be in LossModule.")
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


def get_norm_states(module: nn.Module) -> DictStrAny:
    """Get the state_dict of batch norms in the module.

    Args:
        module (nn.Module): Module to get batch norm states from.
    """
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, _NormBase):
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
        if get_world_size() == 1:
            return
        norm_states = get_norm_states(model)
        if len(norm_states) == 0:
            return
        norm_states = all_reduce_dict(norm_states, reduce_op="mean")
        model.load_state_dict(norm_states, strict=False)


class YOLOXSyncRandomResizeCallback(Callback):
    """Callback for syncing random resize during YOLOX training."""

    def __init__(
        self,
        *args: ArgsType,
        size_list: list[tuple[int, int]],
        interval: int,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback."""
        super().__init__(*args, **kwargs)
        self.size_list = size_list
        self.interval = interval
        self.random_shape = size_list[-1]

    def _get_random_shape(self, device: torch.device) -> tuple[int, int]:
        """Randomly generate shape from size_list and sync across ranks."""
        shape_tensor = torch.zeros(2, dtype=torch.int).to(device)
        if get_rank() == 0:
            random_shape = random.choice(self.size_list)
            shape_tensor[0], shape_tensor[1] = random_shape[0], random_shape[1]
        synchronize()
        shape_tensor = broadcast(shape_tensor, 0)
        return (int(shape_tensor[0].item()), int(shape_tensor[1].item()))

    def on_train_batch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
        batch: DictDataOrList,
        batch_idx: int,
    ) -> None:
        """Hook to run at the start of a training batch."""
        if not isinstance(batch, list):
            batch = [batch]
        if (trainer_state["global_step"] + 1) % self.interval == 0:
            self.random_shape = self._get_random_shape(
                batch[0][K.images].device
            )

        for b in batch:
            scale_y = self.random_shape[0] / b[K.images].shape[-2]
            scale_x = self.random_shape[1] / b[K.images].shape[-1]

            if scale_y == 1 and scale_x == 1:
                return

            # resize images
            b[K.images] = F.interpolate(
                b[K.images],
                size=self.random_shape,
                mode="bilinear",
                align_corners=False,
            )
            b[K.input_hw] = [
                self.random_shape for _ in range(b[K.images].size(0))
            ]

            # resize boxes
            for boxes in b[K.boxes2d]:
                boxes[..., ::2] = boxes[..., ::2] * scale_x
                boxes[..., 1::2] = boxes[..., 1::2] * scale_y
