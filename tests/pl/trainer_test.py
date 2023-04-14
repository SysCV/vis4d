"""Pytorch lightning utilities for unit tests."""
from __future__ import annotations

import shutil
import unittest

from pytorch_lightning import Callback
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.util import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import DataPipe, build_train_dataloader
from vis4d.data.transforms import (
    compose_batch,
    mask,
    normalize,
    pad,
    resize,
    to_tensor,
)
from vis4d.data.transforms.base import compose
from vis4d.data.typing import DictData
from vis4d.engine.connectors import (
    DataConnectionInfo,
    StaticDataConnector,
    data_key,
    pred_key,
)
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss
from vis4d.pl import DefaultTrainer
from vis4d.pl.training_module import TrainingModule


def seg_pipeline(data: list[DictData]) -> DictData:
    """Default data pipeline."""
    return compose_batch([pad.PadImages(value=255), to_tensor.ToTensor()])(
        data
    )


def get_train_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for training."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            resize.ResizeInstanceMasks(),
            normalize.NormalizeImage(),
            mask.ConvertInstanceMaskToSegMask(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe,
        batchprocess_fn=seg_pipeline,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )


def get_trainer(
    exp_name: str, callbacks: None | list[Callback] = None
) -> DefaultTrainer:
    """Build mockup trainer.

    Args:
        exp_name (str): Experiment name
        callbacks (list[Callback], Callback, optional): pl.callbacks that
                                                        should be executed
    """
    if callbacks is None:
        callbacks = []

    return DefaultTrainer(
        work_dir="./unittests/",
        exp_name=exp_name,
        version="test",
        callbacks=callbacks,
        max_steps=2,
        devices=0,
        num_sanity_val_steps=0,
    )


def get_training_module(model: nn.Module):
    """Build mockup training module.

    Args:
        model (nn.Module): Pytorch model
    """
    data_connector = StaticDataConnector(
        connections=DataConnectionInfo(
            train={K.images: K.images},
            test={K.images: K.images},
            loss={
                "output": pred_key("outputs"),
                "target": data_key(K.seg_masks),
            },
        )
    )
    loss_fn = SegCrossEntropyLoss()

    optimizer_cfg = get_optimizer_config(class_config(optim.SGD, lr=0.01))
    return TrainingModule(
        model, [optimizer_cfg], loss_fn, data_connector, seed=1
    )


class PLTrainerTest(unittest.TestCase):
    """Pytorch lightning trainer test class."""

    trainer = get_trainer("test")
    training_module = get_training_module(model=SemanticFPN(num_classes=80))

    def test_train(self) -> None:
        """Test training."""
        dataset = COCO(get_test_data("coco_test"), split="train")
        train_dataloader = get_train_dataloader(dataset, 2)

        self.trainer.fit(self.training_module, train_dataloader)
        shutil.rmtree("./unittests/")
