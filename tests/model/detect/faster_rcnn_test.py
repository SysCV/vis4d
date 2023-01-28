"""Faster RCNN tests."""
from __future__ import annotations

import unittest

import torch
from torch import optim
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
from vis4d.model.detect.faster_rcnn import FasterRCNN, FasterRCNNLoss


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
        """Test inference of Faster RCNN.

        Run::
            >>> pytest vis4d/model/detect/faster_rcnn_test.py::FasterRCNNTest::test_inference # TODO check all these paths!
        """
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

        testcase_gt = torch.load(get_test_file("faster_rcnn.pt"))
        for k in testcase_gt:
            assert k in dets
            for i in range(len(testcase_gt[k])):
                assert (
                    torch.isclose(dets[k][i], testcase_gt[k][i], atol=1e-4)
                    .all()
                    .item()
                )

    def test_train(self) -> None:
        """Test Faster RCNN training."""
        faster_rcnn = FasterRCNN(num_classes=80)
        rcnn_loss = FasterRCNNLoss()

        optimizer = optim.SGD(faster_rcnn.parameters(), lr=0.001, momentum=0.9)

        dataset = COCO(get_test_data("coco_test"), split="train")
        train_loader = get_train_dataloader(dataset, 2, (256, 256))

        running_losses = {}
        faster_rcnn.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids = (
                    data[CommonKeys.images],
                    data[CommonKeys.input_hw],
                    data[CommonKeys.boxes2d],
                    data[CommonKeys.boxes2d_classes],
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = faster_rcnn(
                    inputs, images_hw, gt_boxes, gt_class_ids
                )
                rcnn_losses = rcnn_loss(outputs, images_hw, gt_boxes)
                total_loss = sum(rcnn_losses.values())
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(loss=total_loss, **rcnn_losses)
                for k, loss in losses.items():
                    if k in running_losses:
                        running_losses[k] += loss
                    else:
                        running_losses[k] = loss
                if i % log_step == (log_step - 1):
                    log_str = f"[{epoch + 1}, {i + 1:5d}] "
                    for k, loss in running_losses.items():
                        log_str += f"{k}: {loss / log_step:.3f}, "
                    print(log_str.rstrip(", "))
                    running_losses = {}

    # def test_torchscript(self) -> None: # FIXME
    #     """Test torchscript export of Faster RCNN."""
    #     sample_images = torch.rand((2, 3, 512, 512))
    #     faster_rcnn = FasterRCNN(80)
    #     frcnn_scripted = torch.jit.script(faster_rcnn)
    #     frcnn_scripted(sample_images)
