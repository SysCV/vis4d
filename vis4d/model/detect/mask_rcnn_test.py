"""Mask RCNN tests."""
import unittest

import torch
from torch import optim

from vis4d.data.const import COMMON_KEYS
from vis4d.data.datasets import COCO
from vis4d.op.util import load_model_checkpoint
from vis4d.unittest.util import get_test_file

from .faster_rcnn_test import get_test_dataloader, get_train_dataloader
from .mask_rcnn import REV_KEYS, MaskRCNN, MaskRCNNLoss


class MaskRCNNTest(unittest.TestCase):
    """Mask RCNN test class."""

    def test_inference(self):
        """Test inference of Mask RCNN.

        Run::
            >>> pytest vis4d/model/detect/mask_rcnn_test.py::MaskRCNNTest::test_inference
        """  # pylint: disable=line-too-long # Disable the line length requirement becase of the cmd line prompts
        dataset = COCO(
            get_test_file("coco_test"),
            keys=(COMMON_KEYS.images,),
            split="train",
        )
        test_loader = get_test_dataloader(dataset, 2, (512, 512))
        batch = next(iter(test_loader))
        inputs, images_hw = (
            batch[COMMON_KEYS.images],
            batch[COMMON_KEYS.input_hw],
        )

        weights = (
            "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
            "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
            "20200505_003907-3e542a40.pth"
        )
        mask_rcnn = MaskRCNN(num_classes=80)
        load_model_checkpoint(mask_rcnn, weights, rev_keys=REV_KEYS)

        mask_rcnn.eval()
        with torch.no_grad():
            masks = mask_rcnn(inputs, images_hw, original_hw=images_hw)

        testcase_gt = torch.load(get_test_file("mask_rcnn.pt"))
        for k in testcase_gt:
            assert k in masks
            for i in range(len(testcase_gt[k])):
                assert (
                    torch.isclose(masks[k][i], testcase_gt[k][i], atol=1e-4)
                    .all()
                    .item()
                )

    def test_train(self):
        """Test Mask RCNN training."""
        mask_rcnn = MaskRCNN(num_classes=80)
        mask_rcnn_loss = MaskRCNNLoss(
            mask_rcnn.anchor_gen,
            mask_rcnn.rpn_bbox_encoder,
            mask_rcnn.rcnn_bbox_encoder,
        )

        optimizer = optim.SGD(mask_rcnn.parameters(), lr=0.001, momentum=0.9)

        dataset = COCO(get_test_file("coco_test"), split="train")
        train_loader = get_train_dataloader(dataset, 2, (256, 256))

        running_losses = {}
        mask_rcnn.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids, gt_masks = (
                    data[COMMON_KEYS.images],
                    data[COMMON_KEYS.input_hw],
                    data[COMMON_KEYS.boxes2d],
                    data[COMMON_KEYS.boxes2d_classes],
                    data[COMMON_KEYS.masks],
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = mask_rcnn(inputs, images_hw, gt_boxes, gt_class_ids)
                mask_losses = mask_rcnn_loss(
                    outputs, images_hw, gt_boxes, gt_masks
                )
                total_loss = sum(mask_losses.values())
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(loss=total_loss, **mask_losses)
                for k, v in losses.items():
                    if k in running_losses:
                        running_losses[k] += v
                    else:
                        running_losses[k] = v
                if i % log_step == (log_step - 1):
                    log_str = f"[{epoch + 1}, {i + 1:5d}] "
                    for k, v in running_losses.items():
                        log_str += f"{k}: {v / log_step:.3f}, "
                    print(log_str.rstrip(", "))
                    running_losses = {}
