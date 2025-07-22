"""YOLOX-specific callbacks."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any

import lightning.pytorch as pl
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
from vis4d.op.detect.yolox import YOLOXHeadLoss
from vis4d.op.loss.common import l1_loss

from .base import Callback
from .util import get_loss_module, get_model


class YOLOXModeSwitchCallback(Callback):
    """Callback for switching the mode of YOLOX training."""

    def __init__(
        self, *args: ArgsType, switch_epoch: int, **kwargs: ArgsType
    ) -> None:
        """Init callback.

        Args:
            switch_epoch (int): Epoch to switch the mode.
        """
        super().__init__(*args, **kwargs)
        self.switch_epoch = switch_epoch
        self.switched = False

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the end of a training epoch."""
        if pl_module.current_epoch < self.switch_epoch - 1 or self.switched:
            # TODO: Make work with resume.
            return

        loss_module = get_loss_module(pl_module)

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
        dataloader = trainer.train_dataloader
        assert dataloader is not None
        new_dataloader = DataLoader(
            DataPipe(dataloader.dataset.datasets),
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            sampler=dataloader.sampler,
            persistent_workers=dataloader.persistent_workers,
            pin_memory=dataloader.pin_memory,
        )

        pl_module.check_val_every_n_epoch = 1  # type: ignore

        # Override train_dataloader method in PL datamodule.
        # Set reload_dataloaders_every_n_epochs to 1 to use the new
        # dataloader.
        def train_dataloader() -> DataLoader:  # type: ignore
            """Return dataloader for training."""
            return new_dataloader

        pl_module.datamodule.train_dataloader = train_dataloader  # type: ignore # pylint: disable=line-too-long
        pl_module.reload_dataloaders_every_n_epochs = self.switch_epoch  # type: ignore # pylint: disable=line-too-long

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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Hook to run at the beginning of a testing epoch."""
        if get_world_size() > 1:
            model = get_model(pl_module)
            norm_states = get_norm_states(model)

            if len(norm_states) > 0:
                rank_zero_info("Synced norm states across all processes.")
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

    def on_train_batch_start(  # type: ignore
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Hook to run at the start of a training batch."""
        if not isinstance(batch, list):
            batch = [batch]
        if (trainer.global_step + 1) % self.interval == 0:
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
