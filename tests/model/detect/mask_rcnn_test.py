"""Mask RCNN tests."""
import unittest

import torch
from torch import optim

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets import COCO
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.engine.loss import WeightedMultiLoss
from vis4d.model.detect.mask_rcnn import (
    REV_KEYS,
    MaskDetectionOut,
    MaskRCNN,
    MaskRCNNOut,
)
from vis4d.op.detect.rcnn import (
    MaskRCNNHeadLoss,
    RCNNLoss,
    SampledMaskLoss,
    positive_mask_sampler,
)
from vis4d.op.detect.rpn import RPNLoss

from .faster_rcnn_test import get_test_dataloader, get_train_dataloader


class MaskRCNNTest(unittest.TestCase):
    """Mask RCNN test class."""

    def test_inference(self):
        """Test inference of Mask RCNN.

        Run::
            >>> pytest vis4d/model/detect/mask_rcnn_test.py::MaskRCNNTest::test_inference
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
            "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
            "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
            "20200505_003907-3e542a40.pth"
        )
        mask_rcnn = MaskRCNN(num_classes=80)
        load_model_checkpoint(mask_rcnn, weights, rev_keys=REV_KEYS)

        mask_rcnn.eval()
        with torch.no_grad():
            masks = mask_rcnn(inputs, images_hw, original_hw=images_hw)

        assert isinstance(masks, MaskDetectionOut)

        test_samples = 10

        testcase_gt = torch.load(get_test_file("mask_rcnn.pt"))

        def _assert_eq(
            prediction: torch.Tensor, gts: torch.Tensor, n_samples=test_samples
        ) -> None:
            """Assert prediction and ground truth are equal."""
            for pred, gt in zip(prediction, gts):
                assert (
                    torch.isclose(pred[:n_samples], gt, atol=1e-4).all().item()
                )

        _assert_eq(masks.boxes.boxes, testcase_gt["boxes2d"])
        _assert_eq(masks.boxes.scores, testcase_gt["boxes2d_scores"])
        _assert_eq(masks.boxes.class_ids, testcase_gt["boxes2d_classes"])
        _assert_eq(masks.masks.masks, testcase_gt["masks"])

    def test_train(self):
        """Test Mask RCNN training."""
        mask_rcnn = MaskRCNN(num_classes=80)

        rpn_loss = RPNLoss(
            mask_rcnn.faster_rcnn_heads.anchor_generator,
            mask_rcnn.faster_rcnn_heads.rpn_box_encoder,
        )
        rcnn_loss = RCNNLoss(mask_rcnn.faster_rcnn_heads.rcnn_box_encoder)
        mask_loss = SampledMaskLoss(positive_mask_sampler, MaskRCNNHeadLoss())
        mask_rcnn_loss = WeightedMultiLoss(
            [
                {"loss": rpn_loss, "weight": 1.0},
                {"loss": rcnn_loss, "weight": 1.0},
                {"loss": mask_loss, "weight": 1.0},
            ]
        )

        optimizer = optim.SGD(mask_rcnn.parameters(), lr=0.001, momentum=0.9)

        dataset = COCO(get_test_data("coco_test"), split="train")
        train_loader = get_train_dataloader(dataset, 2, (256, 256))

        running_losses = {}
        mask_rcnn.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids, gt_masks = (
                    data[K.images],
                    data[K.input_hw],
                    data[K.boxes2d],
                    data[K.boxes2d_classes],
                    data[K.instance_masks],
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = mask_rcnn(inputs, images_hw, gt_boxes, gt_class_ids)
                boxes = outputs.boxes
                assert isinstance(outputs, MaskRCNNOut)

                mask_losses = mask_rcnn_loss(
                    cls_outs=boxes.rpn.cls,
                    reg_outs=boxes.rpn.box,
                    images_hw=images_hw,
                    class_outs=boxes.roi.cls_score,
                    regression_outs=boxes.roi.bbox_pred,
                    boxes=boxes.sampled_proposals.boxes,
                    boxes_mask=boxes.sampled_targets.labels,
                    target_boxes=boxes.sampled_targets.boxes,
                    target_classes=boxes.sampled_targets.labels,
                    pred_sampled_proposals=boxes.sampled_proposals,
                    mask_preds=outputs.masks.mask_pred,
                    target_masks=gt_masks,
                    sampled_target_indices=boxes.sampled_target_indices,
                    sampled_targets=boxes.sampled_targets,
                    sampled_proposals=boxes.sampled_proposals,
                )

                total_loss = sum(mask_losses.values())
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = {"loss": total_loss, **mask_losses}
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
