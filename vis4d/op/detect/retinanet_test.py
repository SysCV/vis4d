"""RetinaNet tests."""
import random
import unittest
from typing import Optional, Tuple

import skimage
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from vis4d.common_to_revise.datasets import (
    bdd100k_track_map,
    bdd100k_track_sample,
)
from vis4d.data_to_revise.utils import transform_bbox
from vis4d.op.utils import load_model_checkpoint
from vis4d.struct_to_revise import Boxes2D

from ..base.resnet import ResNet
from ..fpp.fpn import FPN
from .faster_rcnn_test import (
    identity_collate,
    normalize,
    url_to_tensor,
    SampleDataset,
)
from .retinanet import Dense2Det, RetinaNetHead

REV_KEYS = [
    (r"^bbox_head\.", ""),
    (r"^backbone\.", "body."),
    (r"^neck.lateral_convs\.", "inner_blocks."),
    (r"^neck.fpn_convs\.", "layer_blocks."),
    (r"^layer_blocks.3\.", "extra_blocks.p6."),
    (r"^layer_blocks.4\.", "extra_blocks.p7."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]


class RetinaNetTest(unittest.TestCase):
    """RetinaNet test class."""

    def test_inference(self):
        """Test inference of RetinaNet.

        Run::
            >>> pytest vis4d/op/detect/retinanet_test.py::RetinaNetTest::test_inference
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

        basemodel = ResNet("resnet50", trainable_layers=3)
        fpn = FPN(
            basemodel.out_channels[3:],
            256,
            LastLevelP6P7(2048, 256),
            start_index=3,
        )
        retina_net = RetinaNetHead(num_classes=80, in_channels=256)

        dense2det = Dense2Det(
            retina_net.anchor_generator,
            retina_net.box_encoder,
            num_pre_nms=1000,
            max_per_img=100,
            nms_threshold=0.5,
            score_thr=0.05,
        )

        weights = (
            "mmdet://retinanet/retinanet_r50_fpn_2x_coco/"
            "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
        )
        load_model_checkpoint(basemodel, weights, REV_KEYS)
        load_model_checkpoint(fpn, weights, REV_KEYS)
        load_model_checkpoint(retina_net, weights, REV_KEYS)

        retina_net.eval()
        with torch.no_grad():
            features = fpn(basemodel(sample_images))
            outs = retina_net(features[-5:])
            dets = dense2det(
                class_outs=outs.cls_score,
                regression_outs=outs.bbox_pred,
                images_hw=images_hw,
            )

        from vis4d.vis.image import imshow_bboxes
        imshow_bboxes(
            image1[0], dets.boxes[0], dets.scores[0], dets.class_ids[0]
        )
        imshow_bboxes(
            image2[0], dets.boxes[1], dets.scores[1], dets.class_ids[1]
        )
