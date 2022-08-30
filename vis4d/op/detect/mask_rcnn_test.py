"""Mask RCNN tests."""
import unittest
from typing import List, NamedTuple, Optional, Tuple

import skimage
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

# TODO how to handle category IDs?
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from scalabel.label.typing import ImageSize

from vis4d.common.datasets import bdd100k_track_map, bdd100k_segtrack_sample
from vis4d.data.utils import transform_bbox
from vis4d.op.heads.dense_head.rpn import RPNLoss
from vis4d.op.heads.roi_head.rcnn import (
    MaskRCNNLoss,
    RCNNLoss,
    RoI2Det,
    Det2Mask,
    MaskRCNNHead,
)
from vis4d.op.utils import load_model_checkpoint
from vis4d.struct import Boxes2D, InstanceMasks
from vis4d.struct.labels import Masks
from vis4d.vis.image import imshow_bboxes, imshow_masks

from ..backbone.resnet import ResNet
from .faster_rcnn import (
    FasterRCNN,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from .mask_rcnn import MaskRCNN

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^roi_head.mask_head\.", "mask_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "backbone.fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "backbone.fpn.layer_blocks."),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]

from .testcases.faster_rcnn import (
    DET0_BOXES,
    DET0_CLASS_IDS,
    DET0_SCORES,
    DET1_BOXES,
    DET1_CLASS_IDS,
    DET1_SCORES,
    TOPK_PROPOSAL_BOXES,
)


def normalize(img: torch.Tensor) -> torch.Tensor:
    pixel_mean = (123.675, 116.28, 103.53)
    pixel_std = (58.395, 57.12, 57.375)
    pixel_mean = torch.tensor(pixel_mean, device=img.device).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=img.device).view(-1, 1, 1)
    img = (img - pixel_mean) / pixel_std
    return img


def url_to_tensor(
    url: str, im_wh: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    image = skimage.io.imread(url)
    if im_wh is not None:
        image = skimage.transform.resize(image, im_wh) * 255
    return normalize(
        torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).contiguous()
    )


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
        return img, labels.boxes, labels.class_ids, masks.masks


def identity_collate(batch):
    return tuple(zip(*batch))


class MaskRCNNTest(unittest.TestCase):
    """Mask RCNN test class."""

    def test_inference(self):
        """Test inference of Mask RCNN."""
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])

        backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)

        mask_rcnn = MaskRCNN(num_classes=80)

        weights = (
            "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
            "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
            "20200505_003907-3e542a40.pth"
        )

        load_model_checkpoint(backbone, weights, REV_KEYS)
        load_model_checkpoint(mask_rcnn.faster_rcnn, weights, REV_KEYS)
        load_model_checkpoint(mask_rcnn, weights, REV_KEYS)

        mask_rcnn.eval()
        with torch.no_grad():
            features = backbone(sample_images)
            # features = fpn(features)
            outs = mask_rcnn(features)
            roi2det = RoI2Det(
                mask_rcnn.faster_rcnn.rcnn_box_encoder, score_threshold=0.5
            )
            det2mask = Det2Mask(mask_threshold=0.5)
            dets = roi2det(
                class_outs=outs.frcnn_out.roi.cls_score,
                regression_outs=outs.frcnn_out.roi.bbox_pred,
                boxes=outs.frcnn_out.proposals.boxes,
                images_shape=sample_images.shape,
            )
            masks = det2mask(
                mask_outs=outs.roi_mask_out,
                dets=dets,
                images_shape=sample_images.shape,
            )

        imshow_masks(
            image1[0],
            InstanceMasks(
                masks.masks[0], masks.class_ids[0], score=masks.scores[0]
            ),
        )

        # _, topk = torch.topk(outs.proposal_scores[0], 100)
        # assert outs.proposal_boxes[0][topk].shape[0] == 100
        # assert outs.proposal_boxes[0][topk].shape.numel() == 400
        # assert (
        #     torch.isclose(outs.proposal_boxes[0][topk], TOPK_PROPOSAL_BOXES)
        #     .all()
        #     .item()
        # )
        # assert torch.isclose(dets[0].boxes, DET0_BOXES).all().item()
        # assert (
        #     torch.isclose(dets[0].scores, DET0_SCORES, atol=1e-4).all().item()
        # )
        # assert torch.equal(dets[0].class_ids, DET0_CLASS_IDS)
        # assert torch.isclose(dets[1].boxes, DET1_BOXES).all().item()
        # assert (
        #     torch.isclose(dets[1].scores, DET1_SCORES, atol=1e-4).all().item()
        # )
        # assert torch.equal(dets[1].class_ids, DET1_CLASS_IDS)

        # imshow_bboxes(image1[0], *dets[0])
        # imshow_bboxes(image2[0], *dets[1])

    def test_train(self):
        """Test Mask RCNN training."""
        # TODO should bn be frozen during training?
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        mask_rcnn = MaskRCNN(
            num_classes=8,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=8)
        mask_rcnn_loss = MaskRCNNLoss()

        optimizer = optim.SGD(mask_rcnn.parameters(), lr=0.001, momentum=0.9)

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        running_losses = {}
        mask_rcnn.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, gt_boxes, gt_class_ids, gt_masks = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                features = backbone(inputs)
                outputs = mask_rcnn(
                    features, inputs.shape, gt_boxes, gt_class_ids, gt_masks
                )
                rpn_losses = rpn_loss(
                    outputs.rpn_cls_out,
                    outputs.rpn_reg_out,
                    gt_boxes,
                    gt_class_ids,
                    inputs.shape,
                )
                rcnn_losses = rcnn_loss(
                    outputs.roi_cls_out,
                    outputs.roi_reg_out,
                    outputs.proposal_boxes,
                    outputs.proposal_labels,
                    outputs.proposal_target_boxes,
                    outputs.proposal_target_classes,
                )
                mask_losses = mask_rcnn_loss(
                    outputs.roi_mask_out,
                    outputs.proposal_boxes,
                    outputs.proposal_labels,
                    outputs.proposal_masks,
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

    def test_torchscript(self):
        """Test torchscript export of Faster RCNN."""
        sample_images = torch.rand((2, 3, 512, 512))
        backbone = (
            TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
        )
        faster_rcnn = FasterRCNN()
        backbone_scripted = torch.jit.script(backbone)
        frcnn_scripted = torch.jit.script(faster_rcnn)
        features = backbone_scripted(sample_images)
        frcnn_scripted(features, sample_images.shape)


if __name__ == "__main__":
    test = MaskRCNNTest()
    test.test_inference()
