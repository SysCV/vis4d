import unittest
from typing import List, NamedTuple

import skimage
import torch

from vis4d.model.heads.roi_head.rcnn import TransformMMDetFRCNNRoIHeadOutputs
from vis4d.struct import Boxes2D
from vis4d.vis.image import imshow_bboxes

from .faster_rcnn import FasterRCNN


def normalize(img: torch.Tensor) -> torch.Tensor:
    pixel_mean = (123.675, 116.28, 103.53)
    pixel_std = (58.395, 57.12, 57.375)
    pixel_mean = torch.tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std).view(-1, 1, 1)
    img = (img - pixel_mean) / pixel_std
    return img


def url_to_tensor(url: str) -> torch.Tensor:
    image = skimage.io.imread(url)
    image_resized = skimage.transform.resize(image, (512, 512)) * 255
    return normalize(
        torch.tensor(image_resized)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


class Proposals(NamedTuple):
    boxes: List[torch.Tensor]  # N, 4
    scores: List[torch.Tensor]


class Detection(NamedTuple):
    boxes: torch.Tensor  # N, 4
    scores: torch.Tensor
    class_ids: torch.Tensor


class Track2D(NamedTuple):
    boxes: torch.Tensor  # N, 4
    scores: torch.Tensor
    class_ids: torch.Tensor
    track_ids: torch.Tensor


class FasterRCNNTest(unittest.TestCase):
    def test_inference(self):
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg"
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg"
        )
        sample_images = torch.cat([image1, image2])
        faster_rcnn = FasterRCNN(
            weights="mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        )
        faster_rcnn.eval()
        with torch.no_grad():
            outs = faster_rcnn(sample_images)
            transform_outs = TransformMMDetFRCNNRoIHeadOutputs(
                score_threshold=0.5
            )
            dets = transform_outs(
                class_outs=outs.roi_cls_out,
                regression_outs=outs.roi_reg_out,
                boxes=outs.proposal_boxes,
                images_shape=sample_images.shape,
            )

        _, topk = torch.topk(outs.proposal_scores[0], 100)
        imshow_bboxes(image1[0], Boxes2D(outs.proposal_boxes[0][topk]))
        imshow_bboxes(image1[0], dets[0])
        imshow_bboxes(image2[0], dets[1])

    def test_torchscript(self):
        sample_images = torch.rand((2, 3, 512, 512))
        faster_rcnn = FasterRCNN(
            weights="mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        )
        frcnn_scripted = torch.jit.script(faster_rcnn)
        frcnn_scripted(sample_images)
