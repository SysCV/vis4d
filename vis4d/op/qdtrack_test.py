"""QDTrack test file."""
import unittest

import torch
import torch.optim as optim
from mmcv.runner.checkpoint import load_checkpoint
from torch.utils.data import DataLoader

from vis4d.op.detect.faster_rcnn_test import (
    FasterRCNNHead,
    ResNet,
    RoI2Det,
    SampleDataset,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
    identity_collate,
)
from vis4d.op.heads.dense_head.rpn import RPNLoss
from vis4d.op.heads.roi_head.rcnn import RCNNLoss, RoI2Det
from vis4d.op.qdtrack.qdtrack import QDTrack


def pad(images: torch.Tensor, stride=32) -> torch.Tensor:
    """Pad image tensor to be compatible with stride."""
    N, C, H, W = images.shape
    pad = lambda x: (x + (stride - 1)) // stride * stride
    pad_hw = pad(H), pad(W)
    padded_imgs = images.new_zeros((N, C, *pad_hw))
    padded_imgs[:, :, :H, :W] = images
    return padded_imgs


REV_KEYS = [
    (r"^detector.rpn_head.mm_dense_head\.", "rpn_head."),
    ("\.rpn_reg\.", ".rpn_box."),
    (r"^detector.roi_head.mm_roi_head.bbox_head\.", "roi_head."),
    (r"^detector.backbone.mm_backbone\.", "backbone.body."),
    (
        r"^detector.backbone.neck.mm_neck.lateral_convs\.",
        "backbone.fpn.inner_blocks.",
    ),
    (
        r"^detector.backbone.neck.mm_neck.fpn_convs\.",
        "backbone.fpn.layer_blocks.",
    ),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]


class QDTrackTest(unittest.TestCase):
    """QDTrack class tests."""

    def test_inference(self):
        """Inference test."""
        backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        faster_rcnn = FasterRCNNHead(num_classes=8)
        transform_detections = RoI2Det(
            faster_rcnn.rcnn_box_encoder, score_threshold=0.05
        )
        qdtrack = QDTrack()

        load_checkpoint(
            backbone,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        load_checkpoint(
            faster_rcnn,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        load_checkpoint(
            qdtrack,
            "./qdtrack_r50_65point7.ckpt",
            map_location=torch.device("cpu"),
            revise_keys=REV_KEYS,
        )

        qdtrack.eval()
        test_data = SampleDataset()

        batch_size = 2
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=identity_collate,
        )

        with torch.no_grad():
            for data in test_loader:
                inputs, _, _, _, frame_ids = data
                images = pad(torch.cat(inputs))

                features = backbone(images)
                detector_out = faster_rcnn(features)
                boxes, scores, class_ids = transform_detections(
                    *detector_out.roi,
                    detector_out.proposals.boxes,
                    images.shape,
                )
                from vis4d.vis.image import imshow_bboxes

                for img, boxs, score, cls_id in zip(
                    images, boxes, scores, class_ids
                ):
                    imshow_bboxes(img, boxs, score, cls_id)

                outs = qdtrack(features, boxes, scores, class_ids, frame_ids)
                # TODO copy _forward_test code here

                for img, out in zip(images, outs):
                    track_ids, boxes, scores, class_ids, _ = out
                    imshow_bboxes(img, boxes, scores, class_ids, track_ids)

    def test_train(self):
        """Training test."""
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        faster_rcnn = FasterRCNNHead(
            num_classes=8,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=8)
        qdtrack = QDTrack()

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
                inputs, gt_boxes, gt_class_ids, gt_track_ids, _ = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                features = backbone(inputs)
                detector_out = faster_rcnn(features, gt_boxes, gt_class_ids)

                # TODO detector losses only on keyframes
                rpn_losses = rpn_loss(
                    detector_out.rpn.cls,
                    detector_out.rpn.box,
                    gt_boxes,
                    gt_class_ids,
                    inputs.shape,
                )
                rcnn_losses = rcnn_loss(
                    detector_out.roi.cls_score,
                    detector_out.roi.bbox_pred,
                    detector_out.sampled_proposals.boxes,
                    detector_out.sampled_targets.labels,
                    detector_out.sampled_targets.boxes,
                    detector_out.sampled_targets.classes,
                )

                track_losses = qdtrack(
                    features,
                    detector_out.proposals.boxes,
                    detector_out.proposals.scores,
                    None,
                    None,
                    gt_boxes,
                    gt_track_ids,
                )
                total_loss = sum((*rpn_losses, *rcnn_losses, *track_losses))
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(
                    loss=total_loss,
                    **rpn_losses._asdict(),
                    **rcnn_losses._asdict(),
                    **track_losses._asdict(),
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
