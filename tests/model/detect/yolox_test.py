"""YOLOX tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.data.loader import DataPipe, build_inference_dataloaders
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import GenerateResizeParameters, ResizeImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.model.detect.yolox import YOLOX
from vis4d.op.detect.common import DetOut

import multiprocessing

multiprocessing.set_start_method("fork")


def get_test_dataloader(
    datasets: Dataset, batch_size: int, im_hw: tuple[int, int]
) -> DataLoader:
    """Get data loader for testing."""
    preprocess_fn = compose(
        [
            GenerateResizeParameters(
                im_hw, keep_ratio=True, align_long_edge=True
            ),
            ResizeImages(),
        ]
    )
    batchprocess_fn = compose([PadImages(pad2square=True), ToTensor()])
    datapipe = DataPipe(datasets, preprocess_fn)
    return build_inference_dataloaders(
        datapipe,
        samples_per_gpu=batch_size,
        workers_per_gpu=1,
        batchprocess_fn=batchprocess_fn,
    )[0]


class YOLOXTest(unittest.TestCase):
    """YOLOX test class."""

    def test_inference(self) -> None:
        """Test inference of YOLOX."""
        dataset = COCO(
            get_test_data("coco_test"),
            keys_to_load=(K.images,),
            split="train",
            image_channel_mode="BGR",
        )
        test_loader = get_test_dataloader(dataset, 2, (640, 640))
        batch = next(iter(test_loader))
        inputs, images_hw = (batch[K.images], batch[K.input_hw])

        yolox = YOLOX(num_classes=80, weights="mmdet")

        yolox.eval()
        with torch.no_grad():
            dets = yolox(inputs, images_hw, original_hw=images_hw)
        assert isinstance(dets, DetOut)

        testcase_gt = torch.load(get_test_file("yolox.pt"))

        def _assert_eq(
            prediction: list[torch.Tensor], gts: list[torch.Tensor]
        ) -> None:
            """Assert prediction and ground truth are equal."""
            for pred, gt in zip(prediction, gts):
                assert torch.isclose(pred, gt, atol=1e-3).all().item()

        _assert_eq(dets.boxes, testcase_gt.boxes)
        _assert_eq(dets.scores, testcase_gt.scores)
        _assert_eq(dets.class_ids, testcase_gt.class_ids)
