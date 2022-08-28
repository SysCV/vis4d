"""Faster RCNN tests."""
import unittest
from typing import List, NamedTuple, Optional, Tuple

import skimage
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

# TODO how to handle category IDs?
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from vis4d.common.datasets import bdd100k_track_map, bdd100k_track_sample
from vis4d.data.utils import transform_bbox
from vis4d.model.heads.dense_head.rpn import RPNLoss
from vis4d.model.heads.roi_head.rcnn import RCNNLoss, TransformRCNNOutputs
from vis4d.struct import Boxes2D
from vis4d.vis.image import imshow_bboxes

from .faster_rcnn import (
    FasterRCNN,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
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


class TorchResNetBackbone(nn.Module):
    """
    @fyu Leave it here for now. We will move it to a separate file later.
    """

    def __init__(
        self, name: str, pretrained: bool = True, trainable_layers: int = 3
    ):
        super().__init__()
        self.backbone = resnet_fpn_backbone(
            name, pretrained=pretrained, trainable_layers=trainable_layers
        )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(images).values())


class SampleDataset(Dataset):
    def __init__(
        self,
        return_frame_id: bool = False,
        im_wh: Optional[Tuple[int, int]] = None,
    ):
        self.return_frame_id = return_frame_id
        self.im_wh = im_wh
        self.scalabel_data = bdd100k_track_sample()

    def __len__(self):
        return len(self.scalabel_data.frames)

    def __getitem__(self, item):
        frame = self.scalabel_data.frames[item]
        img = url_to_tensor(frame.url, im_wh=self.im_wh)
        labels = Boxes2D.from_scalabel(frame.labels, bdd100k_track_map)
        if self.im_wh is not None:
            trans_mat = torch.eye(3)
            trans_mat[0, 0] = self.im_wh[0] / img.size(3)
            trans_mat[1, 1] = self.im_wh[1] / img.size(2)
            labels.boxes[:, :4] = transform_bbox(
                trans_mat, labels.boxes[:, :4]
            )
        if self.return_frame_id:
            return img, labels.boxes, labels.class_ids, frame.frameIndex - 165
        else:
            return img, labels.boxes, labels.class_ids


def identity_collate(batch):
    return tuple(zip(*batch))


class FasterRCNNTest(unittest.TestCase):
    """Faster RCNN test class."""

    def test_inference(self):
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])

        faster_rcnn = FasterRCNN(
            backbone=TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
            num_classes=80,
            weights=(
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/"
                "faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_"
                "20200504_210434-a5d8aa15.pth"
            ),
        )
        faster_rcnn.eval()
        with torch.no_grad():
            outs = faster_rcnn(sample_images)
            transform_outs = TransformRCNNOutputs(
                faster_rcnn.rcnn_box_encoder, score_threshold=0.5
            )
            dets = transform_outs(
                class_outs=outs.roi_cls_out,
                regression_outs=outs.roi_reg_out,
                boxes=outs.proposal_boxes,
                images_shape=sample_images.shape,
            )

        _, topk = torch.topk(outs.proposal_scores[0], 100)
        imshow_bboxes(image1[0], outs.proposal_boxes[0][topk])
        imshow_bboxes(image1[0], *dets[0])
        imshow_bboxes(image2[0], *dets[1])

    def test_train(self):
        # TODO should bn be frozen during training?
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        faster_rcnn = FasterRCNN(
            TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
            num_classes=8,
        )
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=8)

        optimizer = optim.SGD(faster_rcnn.parameters(), lr=0.001, momentum=0.9)

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        running_losses = {}
        faster_rcnn.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, gt_boxes, gt_class_ids = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = faster_rcnn(inputs, gt_boxes, gt_class_ids)
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
        sample_images = torch.rand((2, 3, 512, 512))
        faster_rcnn = FasterRCNN(
            backbone=TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
            weights=(
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/"
                "faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_"
                "20200504_210434-a5d8aa15.pth"
            ),
        )
        frcnn_scripted = torch.jit.script(faster_rcnn)
        frcnn_scripted(sample_images)
