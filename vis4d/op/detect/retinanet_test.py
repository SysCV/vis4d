"""RetinaNet tests."""
import unittest

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from vis4d.data_to_revise.utils import transform_bbox
from vis4d.op.utils import load_model_checkpoint
from vis4d.struct_to_revise import Boxes2D

from ..base.resnet import ResNet
from ..fpp.fpn import FPN
from .faster_rcnn_test import (
    SampleDataset,
    identity_collate,
    normalize,
    url_to_tensor,
)
from .retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetLoss,
    get_default_box_matcher,
    get_default_box_sampler,
)

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

        basemodel.eval()
        fpn.eval()
        retina_net.eval()
        with torch.no_grad():
            features = fpn(basemodel(sample_images))
            outs = retina_net(features[-5:])
            dets = dense2det(
                class_outs=outs.cls_score,
                regression_outs=outs.bbox_pred,
                images_hw=images_hw,
            )

        # from vis4d.vis.image import imshow_bboxes
        # imshow_bboxes(
        #     image1[0], dets.boxes[0], dets.scores[0], dets.class_ids[0]
        # )
        # imshow_bboxes(
        #     image2[0], dets.boxes[1], dets.scores[1], dets.class_ids[1]
        # )

    def test_train(self):
        """Test RetinaNet training."""
        # TODO should bn be frozen during training?
        num_classes = 8
        basemodel = ResNet("resnet50", trainable_layers=3)
        fpn = FPN(
            basemodel.out_channels[3:],
            256,
            LastLevelP6P7(2048, 256),
            start_index=3,
        )
        retina_net = RetinaNetHead(num_classes=num_classes, in_channels=256)
        retinanet_loss = RetinaNetLoss(
            retina_net.anchor_generator,
            retina_net.box_encoder,
            get_default_box_matcher(),
            get_default_box_sampler(),
            torchvision.ops.sigmoid_focal_loss,
        )

        optimizer = optim.SGD(
            [
                *basemodel.parameters(),
                *fpn.parameters(),
                *retina_net.parameters(),
            ],
            lr=0.001,
            momentum=0.9,
        )

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        basemodel.train()
        fpn.train()
        retina_net.train()

        running_losses = {}
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids, _, _ = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                features = fpn(basemodel(inputs))
                outputs = retina_net(features[-5:])
                retinanet_losses = retinanet_loss(
                    outputs.cls_score,
                    outputs.bbox_pred,
                    gt_boxes,
                    images_hw,
                    gt_class_ids,
                )
                total_loss = sum(retinanet_losses)
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = dict(loss=total_loss, **retinanet_losses._asdict())
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
