"""Mask RCNN tests."""
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from scalabel.label.typing import ImageSize
from torch import optim
from torch.utils.data import DataLoader, Dataset

from vis4d.op.box.box2d import apply_mask
from vis4d.op.detect.rcnn import (
    Det2Mask,
    DetOut,
    MaskRCNNHead,
    MaskRCNNLoss,
    RCNNLoss,
    RoI2Det,
)
from vis4d.op.detect.rpn import RPNLoss
from vis4d.op.mask.util import nhw_to_hwc_mask
from vis4d.op.utils import load_model_checkpoint
from vis4d.run.data.datasets import (  # bdd100k_panseg_map,
    bdd100k_panseg_sample,
)
from vis4d.struct_to_revise import Boxes2D
from vis4d.struct_to_revise.labels import Masks

from ..base.resnet import ResNet
from ..detect.faster_rcnn import (
    FasterRCNNHead,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from ..detect.faster_rcnn_test import (
    identity_collate,
    normalize,
    url_to_tensor,
)
from ..detect.mask_rcnn_test import MASK_REV_KEYS, REV_KEYS
from ..fpp.fpn import FPN
from .panoptic_fpn_head import (
    PanopticFPNHead,
    PanopticFPNLoss,
    postprocess_segms,
)
from .simple_fusion_head import SimplePanopticFusionHead

cname = "conv_upsample_layers"
PAN_REV_KEYS = [
    (r"^semantic_head.conv_upsample_layers\.", "conv_upsample_layers."),
    (r"^semantic_head.conv_logits\.", "conv_logits."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
    (rf"^{cname}.0.conv.0.weight", f"{cname}.0.conv.0.0.weight"),
    (rf"^{cname}.0.conv.0.gn\.", f"{cname}.0.conv.0.1."),
]
for l in range(4):
    for i in range(l):
        PAN_REV_KEYS += [
            (
                rf"^{cname}.{l}.conv.{i}.weight",
                f"{cname}.{l}.conv.{i}.0.weight",
            ),
            (rf"^{cname}.{l}.conv.{i}.gn\.", f"{cname}.{l}.conv.{i}.1."),
        ]

# thing classes before stuff classes
bdd100k_panseg_map = {
    "person": 0,
    "rider": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "caravan": 5,
    "motorcycle": 6,
    "trailer": 7,
    "train": 8,
    "truck": 9,
    "dynamic": 10,
    "ego vehicle": 11,
    "ground": 12,
    "static": 13,
    "parking": 14,
    "rail track": 15,
    "road": 16,
    "sidewalk": 17,
    "bridge": 18,
    "building": 19,
    "fence": 20,
    "garage": 21,
    "guard rail": 22,
    "tunnel": 23,
    "wall": 24,
    "banner": 25,
    "billboard": 26,
    "lane divider": 27,
    "parking sign": 28,
    "pole": 29,
    "polegroup": 30,
    "street light": 31,
    "traffic cone": 32,
    "traffic device": 33,
    "traffic light": 34,
    "traffic sign": 35,
    "traffic sign frame": 36,
    "terrain": 37,
    "vegetation": 38,
    "sky": 39,
}


def pad(images: torch.Tensor, stride=32) -> torch.Tensor:
    """Pad image tensor to be compatible with stride."""
    N, C, H, W = images.shape
    pad = lambda x: (x + (stride - 1)) // stride * stride
    pad_hw = pad(H), pad(W)
    padded_imgs = images.new_zeros((N, C, *pad_hw))
    padded_imgs[:, :, :H, :W] = images
    return padded_imgs


class SampleDataset(Dataset):
    def __init__(self, im_wh: Optional[Tuple[int, int]] = None):
        self.im_wh = im_wh
        self.scalabel_data = bdd100k_panseg_sample()

    def __len__(self):
        return len(self.scalabel_data.frames)

    def __getitem__(self, item):
        frame = self.scalabel_data.frames[item]
        img = url_to_tensor(frame.url, im_wh=self.im_wh)
        # labels = Boxes2D.from_scalabel(frame.labels, bdd100k_panseg_map)
        masks = Masks.from_scalabel(
            frame.labels,
            bdd100k_panseg_map,
            image_size=ImageSize(width=1280, height=720),
        )
        labels = masks.get_boxes2d()
        return (
            img,
            (img.shape[2], img.shape[3]),
            labels.boxes,
            labels.class_ids,
            masks.masks,
        )


class PanopticFPNTest(unittest.TestCase):
    """Panoptic FPN test class."""

    def test_inference(self):
        """Test inference of Panoptic FPN.

        Run::
            >>> pytest vis4d/op/panoptic/panoptic_fpn_test.py::PanopticFPNTest::test_inference
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
        faster_rcnn = FasterRCNNHead()
        mask_head = MaskRCNNHead()
        panoptic_head = PanopticFPNHead()
        fusion_head = SimplePanopticFusionHead()

        roi2det = RoI2Det(faster_rcnn.rcnn_box_encoder, score_threshold=0.5)
        det2mask = Det2Mask(mask_threshold=0.5)

        weights = (
            "mmdet://panoptic_fpn/panoptic_fpn_r50_fpn_mstrain_3x_coco/"
            "panoptic_fpn_r50_fpn_mstrain_3x_coco_20210824_171155-5650f98b.pth"
        )

        for module in [basemodel, fpn, faster_rcnn, mask_head, panoptic_head]:
            if isinstance(module, PanopticFPNHead):
                load_model_checkpoint(module, weights, PAN_REV_KEYS)
            elif isinstance(module, MaskRCNNHead):
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
            seg_outs = panoptic_head(features[2:-1])
            post_segs = postprocess_segms(seg_outs, images_hw, images_hw)
            pan_outs = fusion_head(masks, post_segs)

        # from vis4d.vis.image import imshow_masks
        # imshow_masks(image1[0], pan_outs[0])
        # imshow_masks(image2[0], pan_outs[1])

    def test_train(self):
        """Test Panoptic FPN training."""
        num_thing_classes, num_stuff_classes = 10, 30
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        basemodel = ResNet("resnet50", pretrained=True, trainable_layers=3)
        fpn = FPN(basemodel.out_channels[2:], 256)
        faster_rcnn = FasterRCNNHead(
            num_classes=num_thing_classes,
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        mask_head = MaskRCNNHead(num_classes=num_thing_classes)
        panoptic_head = PanopticFPNHead(num_classes=num_stuff_classes)
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=num_thing_classes)
        mask_rcnn_loss = MaskRCNNLoss()
        pan_fpn_loss = PanopticFPNLoss(num_thing_classes, num_stuff_classes)

        optimizer = optim.SGD(
            [
                *basemodel.parameters(),
                *faster_rcnn.parameters(),
                *mask_head.parameters(),
                *panoptic_head.parameters(),
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
        panoptic_head.train()

        running_losses = {}
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids, gt_masks = data
                inputs = pad(torch.cat(inputs))

                # filter boxes for only fg ones
                gt_boxes_fg, gt_class_fg = [], []
                for gt_box, gt_class_id in zip(gt_boxes, gt_class_ids):
                    fg_mask = gt_class_id < num_thing_classes
                    gt_boxes_fg.append(gt_box[fg_mask])
                    gt_class_fg.append(gt_class_id[fg_mask])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                features = fpn(basemodel(inputs))
                outputs = faster_rcnn(
                    features, images_hw, gt_boxes_fg, gt_class_fg
                )
                mask_outs = mask_head(
                    features[2:-1], outputs.sampled_proposals.boxes
                )
                seg_outs = F.interpolate(
                    panoptic_head(features[2:-1]), images_hw[0]
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
                pan_losses = pan_fpn_loss(
                    seg_outs,
                    nhw_to_hwc_mask(gt_masks[0], gt_class_ids[0]).unsqueeze(0),
                )
                total_loss = sum(
                    (*rpn_losses, *rcnn_losses, *mask_losses, pan_losses)
                )
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(
                    loss=total_loss,
                    **rpn_losses._asdict(),
                    **rcnn_losses._asdict(),
                    **mask_losses._asdict(),
                    pan_loss=pan_losses,
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
