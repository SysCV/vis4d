"""Engine trainer tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
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
from vis4d.engine.callbacks import LoggingCallback
from vis4d.engine.connectors import DataConnector, data_key, pred_key
from vis4d.engine.trainer import Trainer
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss

from .optim.optimizer_test import get_optimizer


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


def get_test_dataloader(
    datasets: Dataset, batch_size: int = 1
) -> list[DataLoader]:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            resize.GenerateResizeParameters((64, 64)),
            resize.ResizeImage(),
            normalize.NormalizeImage(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )


class EngineTrainerTest(unittest.TestCase):
    """Engine trainer test class."""

    model = SemanticFPN(num_classes=80)
    dataset = COCO(
        get_test_data("coco_test"),
        keys_to_load=[
            K.images,
            K.original_images,
            K.boxes2d_classes,
            K.instance_masks,
        ],
        split="train",
    )
    train_dataloader = get_train_dataloader(dataset, 2)
    test_dataloader = get_test_dataloader(dataset)
    data_connector = DataConnector(
        train={"images": K.images},
        test={"images": K.images, "original_hw": K.original_hw},
        loss={
            "output": pred_key("outputs"),
            "target": data_key(K.seg_masks),
        },
    )

    trainer = Trainer(
        device=torch.device("cpu"),
        num_epochs=2,
        data_connector=data_connector,
        callbacks=[LoggingCallback(refresh_rate=1)],
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    def test_fit(self) -> None:
        """Test trainer training."""
        optimizers = get_optimizer()
        loss = SegCrossEntropyLoss()

        self.trainer.fit(self.model, optimizers, loss)

        # TODO: add callback to check loss

    def test_test(self) -> None:
        """Test trainer testing."""
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch.manual_seed(0).get_state())

        self.trainer.test(self.model)

        torch.random.set_rng_state(state)
