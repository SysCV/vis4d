"""Engine train tests."""
from __future__ import annotations

import unittest

from torch import optim
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data
from vis4d.common.callbacks import LoggingCallback
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
from vis4d.engine.opt import Optimizer
from vis4d.engine.train import Trainer
from vis4d.model.segment.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegmentCrossEntropyLoss


def segment_pipeline(data: list[DictData]) -> DictData:
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
            mask.ConvertInstanceMaskToSegmentationMask(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe,
        batchprocess_fn=segment_pipeline,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
    )


class EngineTrainTest(unittest.TestCase):
    """Engine train test class."""

    def test_train(self) -> None:
        """Test engine training."""
        model = SemanticFPN(num_classes=80)
        loss_fn = SegmentCrossEntropyLoss()
        optimizer = Optimizer(
            lambda params: optim.SGD(params, lr=0.01, momentum=0.9)
        )
        dataset = COCO(get_test_data("coco_test"), split="train")
        train_loader = get_train_dataloader(dataset, 2)
        data_connector = StaticDataConnector(
            connections=DataConnectionInfo(
                train={K.images: K.images},
                test={K.images: K.images},
                loss={
                    "output": pred_key("outputs"),
                    "target": data_key(K.segmentation_masks),
                },
            )
        )
        callback = {"logger": LoggingCallback(1)}
        trainer = Trainer(2, train_loader, data_connector, callback)

        model.train()
        trainer.train(model, [optimizer], loss_fn)

        # add callback to check loss
