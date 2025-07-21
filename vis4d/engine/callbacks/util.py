"""PyTorch Lightning callbacks utilities."""

from __future__ import annotations


import lightning.pytorch as pl
from torch import nn

from vis4d.engine.loss_module import LossModule
from vis4d.engine.training_module import TrainingModule


def get_model(model: pl.LightningModule) -> nn.Module:
    """Get model from pl module."""
    if isinstance(model, TrainingModule):
        return model.model
    return model


def get_loss_module(loss_module: pl.LightningModule) -> LossModule:
    """Get loss_module from pl module."""
    if isinstance(loss_module, TrainingModule):
        assert loss_module.loss_module is not None
        return loss_module.loss_module
    return loss_module  # type: ignore
