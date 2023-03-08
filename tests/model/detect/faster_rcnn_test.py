"""Faster RCNN tests."""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys
from vis4d.data.datasets import COCO
from vis4d.data.loader import (
    DataPipe,
    build_inference_dataloaders,
    build_train_dataloader,
)
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import normalize_image
from vis4d.data.transforms.pad import pad_image
from vis4d.data.transforms.resize import (
    resize_boxes2d,
    resize_image,
    resize_masks,
)
from vis4d.model.detect.faster_rcnn import FasterRCNN
from vis4d.op.detect.rcnn import DetOut


def get_train_dataloader(
    datasets: Dataset,
    batch_size: int,
    im_hw: tuple[int, int],
    with_mask: bool = False,
) -> DataLoader:
    """Get data loader for training."""
    resize_trans = [resize_image(im_hw, keep_ratio=True), resize_boxes2d()]
    if with_mask:
        resize_trans += [resize_masks()]
    preprocess_fn = compose([*resize_trans, normalize_image()])
    batchprocess_fn = pad_image()
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
    preprocess_fn = compose([resize_image(im_hw), normalize_image()])
    batchprocess_fn = pad_image()
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
        """Test inference of Faster RCNN."""
        dataset = COCO(
            get_test_data("coco_test"),
            keys=(CommonKeys.images,),
            split="train",
        )
        test_loader = get_test_dataloader(dataset, 2, (512, 512))
        batch = next(iter(test_loader))
        inputs, images_hw = (
            batch[CommonKeys.images],
            batch[CommonKeys.input_hw],
        )

        faster_rcnn = FasterRCNN(num_classes=80, weights="mmdet")

        faster_rcnn.eval()
        with torch.no_grad():
            dets = faster_rcnn(inputs, images_hw, original_hw=images_hw)
        assert isinstance(dets, DetOut)

        # TODO: update test gt after refactoring config
        # testcase_gt = torch.load(get_test_file("faster_rcnn.pt"))

        # def _assert_eq(
        #     prediction: list[torch.Tensor], gts: list[torch.Tensor]
        # ) -> None:
        #     """Assert prediction and ground truth are equal."""
        #     for pred, gt in zip(prediction, gts):
        #         assert torch.isclose(pred, gt, atol=1e-4).all().item()

        # _assert_eq(dets.boxes, testcase_gt["boxes2d"])
        # _assert_eq(dets.scores, testcase_gt["boxes2d_scores"])
        # _assert_eq(dets.class_ids, testcase_gt["boxes2d_classes"])

    # def test_cli_training(self) -> None:
    #     """Test Faster RCNN training via CLI."""
    # TODO: add test for training after refactoring config
