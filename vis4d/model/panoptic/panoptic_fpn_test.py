"""Panoptic FPN tests (WIP)."""
from __future__ import annotations

import unittest

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

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
from vis4d.unittest.util import get_test_file

from .panoptic_fpn import PanopticFPN


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


class PanopticFPNTest(unittest.TestCase):
    """Panoptic FPN test class."""

    def test_inference(self):
        """Test inference of Panoptic FPN.

        Run::
            >>> pytest vis4d/model/panoptic/panoptic_fpn_test.py::PanopticFPNTest::test_inference
        """
        dataset = COCO(
            get_test_file("coco_test"),
            keys=(CommonKeys.images,),
            split="train",
        )
        test_loader = get_test_dataloader(dataset, 2, (512, 512))
        batch = next(iter(test_loader))
        inputs, images_hw = (
            batch[CommonKeys.images],
            batch[CommonKeys.input_hw],
        )

        panoptic_fpn = PanopticFPN(
            num_things_classes=80, num_stuff_classes=53, weights="mmdet"
        )

        # TODO

    def test_train(self):
        """Test Panoptic FPN training."""
        # TODO
