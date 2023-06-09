"""Pytorch lightning utilities for unit tests."""
from __future__ import annotations

import shutil
import unittest

from ml_collections import ConfigDict
from pytorch_lightning import Callback
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.config import class_config
from vis4d.config.util import get_optimizer_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import DataPipe, build_train_dataloader
from vis4d.data.transforms import (
    compose,
    mask,
    normalize,
    pad,
    resize,
    to_tensor,
)
from vis4d.data.transforms.base import compose
from vis4d.data.typing import DictData
from vis4d.engine.connectors import (
    DataConnector,
    LossConnector,
    data_key,
    pred_key,
)
from vis4d.engine.loss_module import LossModule
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss
from vis4d.pl.trainer import PLTrainer
from vis4d.pl.training_module import TrainingModule


def seg_pipeline(data: list[DictData]) -> DictData:
    """Default data pipeline."""
    return compose([pad.PadImages(value=255), to_tensor.ToTensor()])(data)


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
) -> PLTrainer:
    """Build mockup trainer.

    Args:
        exp_name (str): Experiment name
        callbacks (list[Callback], Callback, optional): pl.callbacks that
                                                        should be executed
    """
    if callbacks is None:
        callbacks = []

    return PLTrainer(
        work_dir="./unittests/",
        exp_name=exp_name,
        version="test",
        callbacks=callbacks,
        max_steps=2,
        devices=0,
        num_sanity_val_steps=0,
    )


def get_training_module(model: ConfigDict):
    """Build mockup training module.

    Args:
        model (nn.Module): Pytorch model
    """
    train_data_connector = DataConnector(key_mapping={K.images: K.images})
    test_data_connector = DataConnector(key_mapping={K.images: K.images})
    loss_module = LossModule(
        {
            "loss": SegCrossEntropyLoss(),
            "connector": LossConnector(
                key_mapping={
                    "output": pred_key("outputs"),
                    "target": data_key(K.seg_masks),
                }
            ),
        }
    )

    optimizer_cfg = get_optimizer_cfg(class_config(optim.SGD, lr=0.01))
    return TrainingModule(
        model=model,
        optimizers=[optimizer_cfg],
        loss_module=loss_module,
        train_data_connector=train_data_connector,
        test_data_connector=test_data_connector,
        seed=1,
    )


class PLTrainerTest(unittest.TestCase):
    """Pytorch lightning trainer test class."""

    def setUp(self) -> None:
        """Setup."""
        self.trainer = get_trainer("test")

        model_cfg = class_config(
            SemanticFPN,
            num_classes=80,
        )

        self.training_module = get_training_module(model=model_cfg)

    def test_train(self) -> None:
        """Test training."""
        dataset = COCO(get_test_data("coco_test"), split="train")
        train_dataloader = get_train_dataloader(dataset, 2)

        self.trainer.fit(self.training_module, train_dataloader)
        shutil.rmtree("./unittests/")
