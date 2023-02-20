"""RetinaNet tests."""
import unittest

import torch
from torch import optim

from tests.util import get_test_data, get_test_file
from vis4d.data.const import CommonKeys as Keys
from vis4d.data.datasets import COCO
from vis4d.engine.ckpt import load_model_checkpoint
from vis4d.model.detect.retinanet import REV_KEYS, RetinaNet, RetinaNetLoss

from .faster_rcnn_test import get_test_dataloader, get_train_dataloader


class RetinaNetTest(unittest.TestCase):
    """RetinaNet test class."""

    def test_inference(self) -> None:
        """Test inference of RetinaNet.

        Run::
            >>> pytest vis4d/model/detect/retinanet_test.py::RetinaNetTest::test_inference
        """
        dataset = COCO(
            get_test_data("coco_test"),
            keys_to_load=(Keys.images, Keys.boxes2d, Keys.boxes2d_classes),
            split="train",
        )
        test_loader = get_test_dataloader(dataset, 2, (512, 512))
        batch = next(iter(test_loader))
        inputs, images_hw = (
            batch[Keys.images],
            batch[Keys.input_hw],
        )

        weights = (
            "mmdet://retinanet/retinanet_r50_fpn_2x_coco/"
            "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
        )
        retina_net = RetinaNet(num_classes=80)
        load_model_checkpoint(retina_net, weights, rev_keys=REV_KEYS)

        retina_net.eval()
        with torch.no_grad():
            dets = retina_net(inputs, images_hw, original_hw=images_hw)

        testcase_gt = torch.load(get_test_file("retinanet.pt"))
        for k in testcase_gt:
            assert k in dets
            for i in range(len(testcase_gt[k])):
                assert (
                    torch.isclose(dets[k][i], testcase_gt[k][i], atol=1e-4)
                    .all()
                    .item()
                )

    def test_train(self) -> None:
        """Test RetinaNet training."""
        retina_net = RetinaNet(num_classes=80)
        retinanet_loss = RetinaNetLoss(
            retina_net.retinanet_head.anchor_generator,
            retina_net.retinanet_head.box_encoder,
            retina_net.retinanet_head.box_matcher,
            retina_net.retinanet_head.box_sampler,
        )

        optimizer = optim.SGD(retina_net.parameters(), lr=0.001, momentum=0.9)

        dataset = COCO(get_test_data("coco_test"), split="train")
        train_loader = get_train_dataloader(dataset, 2, (256, 256))

        running_losses = {}
        retina_net.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, images_hw, gt_boxes, gt_class_ids = (
                    data[Keys.images],
                    data[Keys.input_hw],
                    data[Keys.boxes2d],
                    data[Keys.boxes2d_classes],
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = retina_net(inputs)
                retinanet_losses = retinanet_loss(
                    outputs, images_hw, gt_boxes, gt_class_ids
                )
                total_loss = sum(retinanet_losses.values())
                total_loss.backward()
                optimizer.step()

                # print statistics
                losses = {"loss": total_loss, **retinanet_losses}
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
