"""Engine trainer tests."""
from __future__ import annotations

import shutil
import tempfile
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
    compose,
    mask,
    normalize,
    pad,
    resize,
    to_tensor,
)
from vis4d.data.typing import DictData
from vis4d.engine.callbacks import LoggingCallback
from vis4d.engine.connectors import (
    DataConnector,
    LossConnector,
    data_key,
    pred_key,
)
from vis4d.engine.loss_module import LossModule
from vis4d.engine.trainer import Trainer
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss

from .optim.optimizer_test import get_optimizer


def seg_pipeline(data: list[DictData]) -> DictData:
    """Default data pipeline."""
    return compose([pad.PadImages(value=255), to_tensor.ToTensor()])(data)


def get_train_dataloader(datasets: Dataset, batch_size: int) -> DataLoader:
    """Get data loader for training."""
    preprocess_fn = compose(
        [
            resize.GenResizeParameters((64, 64)),
            resize.ResizeImages(),
            resize.ResizeInstanceMasks(),
            normalize.NormalizeImages(),
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
            resize.GenResizeParameters((64, 64)),
            resize.ResizeImages(),
            normalize.NormalizeImages(),
        ]
    )
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe, samples_per_gpu=batch_size, workers_per_gpu=1
    )


class EngineTrainerTest(unittest.TestCase):
    """Engine trainer test class."""

    def setUp(self) -> None:
        """Set up test."""
        self.test_dir = tempfile.mkdtemp()

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
        train_data_connector = DataConnector(key_mapping={"images": K.images})
        test_data_connector = DataConnector(
            key_mapping={"images": K.images, "original_hw": K.original_hw}
        )

        self.model = SemanticFPN(num_classes=80)

        self.trainer = Trainer(
            device=torch.device("cpu"),
            output_dir=self.test_dir,
            num_epochs=2,
            train_data_connector=train_data_connector,
            test_data_connector=test_data_connector,
            callbacks=[LoggingCallback(refresh_rate=1)],
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
        )

    def tearDown(self) -> None:
        """Tear down test."""
        shutil.rmtree(self.test_dir)

    def test_fit(self) -> None:
        """Test trainer training."""
        optimizers = get_optimizer()
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

        self.trainer.fit(self.model, optimizers, loss_module)

        # TODO: add callback to check loss

    def test_test(self) -> None:
        """Test trainer testing."""
        state = torch.random.get_rng_state()
        torch.random.set_rng_state(torch.manual_seed(0).get_state())

        self.trainer.test(self.model)

        torch.random.set_rng_state(state)
