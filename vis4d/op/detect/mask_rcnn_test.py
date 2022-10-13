"""Mask RCNN tests."""
import unittest
from typing import Optional, Tuple

import torch
from scalabel.label.typing import ImageSize
from torch import optim
from torch.utils.data import DataLoader, Dataset

from vis4d.op.box.box2d import apply_mask
from vis4d.op.detect.rcnn import (
    Det2Mask,
    MaskRCNNHead,
    MaskRCNNHeadLoss,
    RCNNLoss,
    RoI2Det,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.utils import load_model_checkpoint
from vis4d.run.data.datasets import bdd100k_segtrack_sample, bdd100k_track_map
from vis4d.struct_to_revise import Boxes2D
from vis4d.struct_to_revise.labels import Masks

from ..base.resnet import ResNet
from ..fpp.fpn import FPN
from .faster_rcnn import (
    FasterRCNNHead,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from .faster_rcnn_test import identity_collate, normalize, url_to_tensor
from .testcases.mask_rcnn import (
    INSSEG0_CLASS_IDS,
    INSSEG0_INDICES,
    INSSEG0_MASKS,
    INSSEG0_SCORES,
    INSSEG1_CLASS_IDS,
    INSSEG1_INDICES,
    INSSEG1_MASKS,
    INSSEG1_SCORES,
)

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^backbone\.", "body."),
    (r"^neck.lateral_convs\.", "inner_blocks."),
    (r"^neck.fpn_convs\.", "layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]
MASK_REV_KEYS = [
    (r"^roi_head.mask_head\.", "mask_head."),
    (r"^mask_head.convs\.", "convs."),
    (r"^mask_head.upsample\.", "upsample."),
    (r"^mask_head.conv_logits\.", "conv_logits."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class SampleDataset(Dataset):
    def __init__(self, im_wh: Optional[Tuple[int, int]] = None):
        self.im_wh = im_wh
        self.scalabel_data = bdd100k_segtrack_sample()

    def __len__(self):
        return len(self.scalabel_data.frames)

    def __getitem__(self, item):
        frame = self.scalabel_data.frames[item]
        img = url_to_tensor(frame.url, im_wh=self.im_wh)
        labels = Boxes2D.from_scalabel(frame.labels, bdd100k_track_map)
        masks = Masks.from_scalabel(
            frame.labels,
            bdd100k_track_map,
            image_size=ImageSize(width=1280, height=720),
        )
        return (
            img,
            (img.shape[2], img.shape[3]),
            labels.boxes,
            labels.class_ids,
            masks.masks,
        )


class MaskRCNNTest(unittest.TestCase):
    """Mask RCNN test class."""

    def test_inference(self):
        """Test inference of Mask RCNN.

        Run::
            >>> pytest vis4d/op/detect/mask_rcnn_test.py::MaskRCNNTest::test_inference
        """  # pylint: disable=line-too-long # Disable the line length requirement becase of the cmd line prompts
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])
        images_hw = [(512, 512) for _ in range(2)]

        basemodel = ResNet("resnet50", pretrained=True, trainable_layers=3)
        fpn = FPN(basemodel.out_channels[2:], 256)
        faster_rcnn = FasterRCNNHead(num_classes=80)
        mask_head = MaskRCNNHead(num_classes=80)

        roi2det = RoI2Det(faster_rcnn.rcnn_box_encoder, score_threshold=0.5)
        det2mask = Det2Mask(mask_threshold=0.5)

        weights = (
            "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
            "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
            "20200505_003907-3e542a40.pth"
        )

        for module in [basemodel, fpn, faster_rcnn, mask_head]:
            if isinstance(module, MaskRCNNHead):
                load_model_checkpoint(module, weights, MASK_REV_KEYS)
            else:
                load_model_checkpoint(module, weights, REV_KEYS)
            module.eval()

        with torch.no_grad():
            features = fpn(basemodel(sample_images))
            outs = faster_rcnn(features, images_hw)
            dets = roi2det(
                class_outs=outs.roi.cls_score,
                regression_outs=outs.roi.bbox_pred,
                boxes=outs.proposals.boxes,
                images_hw=images_hw,
            )
            mask_outs = mask_head(features[2:-1], dets.boxes)
            masks = det2mask(
                mask_outs=mask_outs.mask_pred.sigmoid(),
                dets=dets,
                images_hw=images_hw,
            )

        assert (
            torch.isclose(masks.masks[0][INSSEG0_INDICES], INSSEG0_MASKS)
            .all()
            .item()
        )
        assert (
            torch.isclose(masks.scores[0], INSSEG0_SCORES, atol=1e-4)
            .all()
            .item()
        )
        assert torch.equal(masks.class_ids[0], INSSEG0_CLASS_IDS)
        assert (
            torch.isclose(masks.masks[1][INSSEG1_INDICES], INSSEG1_MASKS)
            .all()
            .item()
        )
        assert (
            torch.isclose(masks.scores[1], INSSEG1_SCORES, atol=1e-4)
            .all()
            .item()
        )
        assert torch.equal(masks.class_ids[1], INSSEG1_CLASS_IDS)

        # from vis4d.vis.image import imshow_masks
        # imshow_masks(
        #     image1[0], masks.masks[0], masks.scores[0], masks.class_ids[0]
        # )
        # imshow_masks(
        #     image2[0], masks.masks[1], masks.scores[1], masks.class_ids[1]
        # )

    def test_train(self):
        """Test Mask RCNN training."""
        # TODO should bn be frozen during training?
        num_classes = 8
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        basemodel = ResNet("resnet50", pretrained=True, trainable_layers=3)
        fpn = FPN(basemodel.out_channels[2:], 256)
        faster_rcnn = FasterRCNNHead(
            num_classes=num_classes,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        mask_head = MaskRCNNHead(num_classes=num_classes)
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=num_classes)
        mask_rcnn_loss = MaskRCNNHeadLoss()

        optimizer = optim.SGD(
            [
                *basemodel.parameters(),
                *faster_rcnn.parameters(),
                *mask_head.parameters(),
            ],
            lr=0.001,
            momentum=0.9,
        )

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        basemodel.train()
        faster_rcnn.train()
        mask_head.train()

        running_losses = {}
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids, gt_masks = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                features = fpn(basemodel(inputs))
                outputs = faster_rcnn(
                    features, images_hw, gt_boxes, gt_class_ids
                )
                mask_outs = mask_head(
                    features[2:-1], outputs.sampled_proposals.boxes
                )
                rpn_losses = rpn_loss(
                    outputs.rpn.cls, outputs.rpn.box, gt_boxes, images_hw
                )
                rcnn_losses = rcnn_loss(
                    outputs.roi.cls_score,
                    outputs.roi.bbox_pred,
                    outputs.sampled_proposals.boxes,
                    outputs.sampled_targets.labels,
                    outputs.sampled_targets.boxes,
                    outputs.sampled_targets.classes,
                )
                assert outputs.sampled_target_indices is not None
                sampled_masks = apply_mask(
                    outputs.sampled_target_indices, gt_masks
                )[0]
                mask_losses = mask_rcnn_loss(
                    mask_outs.mask_pred,
                    outputs.sampled_proposals.boxes,
                    outputs.sampled_targets.classes,
                    sampled_masks,
                )
                total_loss = sum((*rpn_losses, *rcnn_losses, *mask_losses))
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(
                    loss=total_loss,
                    **rpn_losses._asdict(),
                    **rcnn_losses._asdict(),
                    **mask_losses._asdict(),
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
