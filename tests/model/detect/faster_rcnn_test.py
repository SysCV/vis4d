"""Faster RCNN tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImage,
    ResizeInstanceMasks,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.model.detect.faster_rcnn import FasterRCNN
from vis4d.op.detect.common import DetOut


def get_train_dataloader(
    datasets: Dataset,
    batch_size: int,
    im_hw: tuple[int, int],
    with_mask: bool = False,
) -> DataLoader:
    """Get data loader for training."""
    resize_trans = [
        GenerateResizeParameters(im_hw, keep_ratio=True),
        ResizeImage(),
        ResizeBoxes2D(),
    ]
    if with_mask:
        resize_trans += [ResizeInstanceMasks()]
    preprocess_fn = compose([*resize_trans, NormalizeImage()])
    batchprocess_fn = compose_batch([PadImages(), ToTensor()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_train_dataloader(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        batchprocess_fn=batchprocess_fn,
    )


def get_test_dataloader(
    datasets: Dataset, batch_size: int, im_hw: tuple[int, int]
) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            GenerateResizeParameters(im_hw),
            ResizeImage(),
            ResizeBoxes2D(),
            NormalizeImage(),
        ]
    )
    batchprocess_fn = compose_batch([PadImages(), ToTensor()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        batchprocess_fn=batchprocess_fn,
    )[0]


class FasterRCNNTest(unittest.TestCase):
    """Faster RCNN test class."""

    def test_inference(self) -> None:
        """Test inference of Faster RCNN.

        Run::
            >>> pytest tests/model/detect/faster_rcnn_test.py::FasterRCNNTest::test_inference
        """
        dataset = COCO(
            get_test_data("coco_test"),
            keys_to_load=(K.images, K.boxes2d, K.boxes2d_classes),
            split="train",
        )
        test_loader = get_test_dataloader(dataset, 2, (512, 512))
        batch = next(iter(test_loader))
        inputs, images_hw = (
            batch[K.images],
            batch[K.input_hw],
        )

        weights = (
            "mmdet://faster_rcnn/faster_rcnn_r50_fpn_1x_coco/"
            "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
        )
        faster_rcnn = FasterRCNN(num_classes=80, weights=weights)

        faster_rcnn.eval()
        with torch.no_grad():
            dets = faster_rcnn(inputs, images_hw, original_hw=images_hw)
        assert isinstance(dets, DetOut)

        testcase_gt = torch.load(get_test_file("faster_rcnn.pt"))

        def _assert_eq(
            prediction: list[torch.Tensor], gts: list[torch.Tensor]
        ) -> None:
            """Assert prediction and ground truth are equal."""
            for pred, gt in zip(prediction, gts):
                assert torch.isclose(pred, gt, atol=1e-4).all().item()

        _assert_eq(dets.boxes, testcase_gt["boxes2d"])
        _assert_eq(dets.scores, testcase_gt["boxes2d_scores"])
        _assert_eq(dets.class_ids, testcase_gt["boxes2d_classes"])

    # TODO: add test for training after refactoring config
    # def test_cli_training(self) -> None:
    #     """Test Faster RCNN training via CLI."""
