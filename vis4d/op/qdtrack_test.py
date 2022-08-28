"""QDTrack test file."""
import unittest
from re import T

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from vis4d.op.detect.faster_rcnn_test import (
    FasterRCNN,
    ResNet,
    SampleDataset,
    RoI2Det,
    identity_collate,
)
from vis4d.op.qdtrack.qdtrack import QDTrack

from .utils import load_model_checkpoint

REV_KEYS = [
    (r"^detector.rpn_head.mm_dense_head\.", "detector.rpn_head."),
    ("\.rpn_reg\.", ".rpn_box."),
    (r"^detector.roi_head.mm_roi_head.bbox_head\.", "detector.roi_head."),
    (r"^detector.backbone.mm_backbone\.", "detector.backbone.backbone.body."),
    (
        r"^detector.backbone.neck.mm_neck.lateral_convs\.",
        "detector.backbone.backbone.fpn.inner_blocks.",
    ),
    (
        r"^detector.backbone.neck.mm_neck.fpn_convs\.",
        "detector.backbone.backbone.fpn.layer_blocks.",
    ),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    def test_inference(self):
        """Inference test."""
        faster_rcnn = FasterRCNN(
            backbone=ResNet("resnet50", pretrained=True, trainable_layers=3),
            num_classes=8,
        )
        transform_outs = RoI2Det(
            faster_rcnn.rcnn_box_encoder, score_threshold=0.5
        )
        qdtrack = QDTrack(faster_rcnn, transform_outs)

        from mmcv.runner.checkpoint import load_checkpoint

        load_checkpoint(
            qdtrack,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
            strict=True,
        )

        qdtrack.eval()
        test_data = SampleDataset(return_frame_id=True)

        batch_size = 2
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=identity_collate,
        )

        with torch.no_grad():
            for data in test_loader:
                inputs, _, _, frame_ids = data
                outs = qdtrack(
                    torch.cat(inputs),
                    frame_ids,
                )
                from vis4d.vis.image import imshow_bboxes

                for img, out in zip(inputs, outs):
                    track_ids, boxes, scores, class_ids, _ = out
                    imshow_bboxes(img[0], boxes, scores, class_ids, track_ids)

    def test_train(self):
        """Training test."""
        faster_rcnn = FasterRCNN(
            backbone=ResNet("resnet50", pretrained=True, trainable_layers=3),
            num_classes=8,
        )
        transform_outs = RoI2Det(
            faster_rcnn.rcnn_box_encoder, score_threshold=0.5
        )
        qdtrack = QDTrack(faster_rcnn, transform_outs)

        optimizer = optim.SGD(qdtrack.parameters(), lr=0.001, momentum=0.9)

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        running_losses = {}
        qdtrack.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, gt_boxes, gt_class_ids = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = qdtrack(inputs, gt_boxes, gt_class_ids)
                total_loss = sum((*rpn_losses, *rcnn_losses))
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(
                    loss=total_loss,
                    **rpn_losses._asdict(),
                    **rcnn_losses._asdict(),
                )
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

    def test_torchscript(self):
        """Test torchscipt export."""
        sample_images = torch.rand((2, 3, 512, 512))
        qdtrack = QDTrack()
        qdtrack_scripted = torch.jit.script(qdtrack)
        qdtrack_scripted(sample_images)
