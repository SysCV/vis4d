"""RetinaNet coco training example."""
import argparse
import warnings
from time import perf_counter
from typing import List, Optional, Tuple, Union

import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from tqdm import tqdm

from vis4d.common_to_revise.data_pipelines import default_test, default_train
from vis4d.data.datasets.base import DataKeys
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import HDF5Backend
from vis4d.eval.coco import COCOEvaluator
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.util import bbox_postprocess
from vis4d.op.detect.rcnn import DetOut
from vis4d.op.detect.retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetLoss,
    RetinaNetLosses,
    RetinaNetOut,
    get_default_box_matcher,
    get_default_box_sampler,
)
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim.warmup import LinearLRWarmup

REV_KEYS = [
    (r"^bbox_head\.", "retinanet_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"^fpn.layer_blocks.3\.", "fpn.extra_blocks.p6."),
    (r"^fpn.layer_blocks.4\.", "fpn.extra_blocks.p7."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]

warnings.filterwarnings("ignore")

log_step = 100
num_epochs = 12
batch_size = 2  # 16
learning_rate = 0.02 / 16 * batch_size
train_resolution = (800, 1333)
test_resolution = (800, 1333)
device = torch.device("cuda")


class RetinaNet(nn.Module):
    """RetinaNet wrapper class for checkpointing etc."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(
            self.backbone.out_channels[3:],
            256,
            LastLevelP6P7(2048, 256),
            start_index=3,
        )
        self.retinanet_head = RetinaNetHead(num_classes=80, in_channels=256)
        self.retinanet_loss = RetinaNetLoss(
            self.retinanet_head.anchor_generator,
            self.retinanet_head.box_encoder,
            get_default_box_matcher(),
            get_default_box_sampler(),
            torchvision.ops.sigmoid_focal_loss,
        )
        self.transform_outs = Dense2Det(
            self.retinanet_head.anchor_generator,
            self.retinanet_head.box_encoder,
            num_pre_nms=1000,
            max_per_img=100,
            nms_threshold=0.5,
            score_thr=0.05,
        )

    def forward(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        original_hw: Optional[List[Tuple[int, int]]] = None,
    ) -> Union[RetinaNetLosses, DetOut]:
        """Forward."""
        if target_boxes is not None:
            assert target_classes is not None
            return self._forward_train(
                images, images_hw, target_boxes, target_classes
            )
        return self._forward_test(images, images_hw, original_hw)

    def _forward_train(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> RetinaNetLosses:
        """Forward training stage."""
        features = self.fpn(self.backbone(images))
        outputs = self.retinanet_head(features[-5:])
        losses = self.retinanet_loss(
            outputs.cls_score,
            outputs.bbox_pred,
            target_boxes,
            images_hw,
            target_classes,
        )
        return losses

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        original_hw: List[Tuple[int, int]],
    ) -> DetOut:
        """Forward testing stage."""
        features = self.fpn(self.backbone(images))
        outs = self.retinanet_head(features[-5:])
        dets, scores, class_ids = self.transform_outs(
            class_outs=outs.cls_score,
            regression_outs=outs.bbox_pred,
            images_hw=images_hw,
        )
        for i, boxs in enumerate(dets):
            dets[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        post_dets = DetOut(boxes=dets, scores=scores, class_ids=class_ids)
        return post_dets


## setup model
retinanet = RetinaNet()

optimizer = optim.SGD(
    retinanet.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=0.0001,
)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[8, 11], gamma=0.1
)
warmup = LinearLRWarmup(0.001, 500)

## setup test dataset
data_root = "data/COCO"
test_loader = default_test(
    COCO(data_root, "val2017", HDF5Backend()), 1, test_resolution
)
test_eval = COCOEvaluator(data_root)


@torch.no_grad()
def validation_loop(model):
    """Validate current model with test dataset."""
    model.eval()
    print("Running validation...")
    for _, data in enumerate(tqdm(test_loader[0])):
        images = data[DataKeys.images].to(device)
        original_hw = data[DataKeys.metadata]["original_hw"]
        images_hw = data[DataKeys.metadata]["input_hw"]

        boxes, scores, class_ids = model(
            images, images_hw, original_hw=original_hw
        )

        test_eval.process(
            data,
            {
                "boxes2d": boxes,
                "boxes2d_scores": scores,
                "boxes2d_classes": class_ids,
            },
        )

    _, log_str = test_eval.evaluate("COCO_AP")
    print(log_str)


def training_loop(model):
    """Training loop."""
    train_loader = default_train(
        COCO(data_root, "train2017", HDF5Backend()),
        batch_size,
        train_resolution,
    )

    running_losses = {}
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            tic = perf_counter()
            inputs, inputs_hw, gt_boxes, gt_class_ids = (
                data[DataKeys.images].to(device),
                data[DataKeys.metadata]["input_hw"],
                [b.to(device) for b in data[DataKeys.boxes2d]],
                [b.to(device) for b in data[DataKeys.boxes2d_classes]],
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            losses = model(inputs, inputs_hw, gt_boxes, gt_class_ids)
            total_loss = sum(losses)
            total_loss.backward()

            if epoch == 0 and i < 500:
                for g in optimizer.param_groups:
                    g["lr"] = warmup(i, learning_rate)
            elif epoch == 0 and i == 500:
                for g in optimizer.param_groups:
                    g["lr"] = learning_rate

            optimizer.step()
            toc = perf_counter()

            # print statistics
            losses = dict(time=toc - tic, loss=total_loss, **losses._asdict())
            for k, v in losses.items():
                if k in running_losses:
                    running_losses[k] += v
                else:
                    running_losses[k] = v
            if i % log_step == (log_step - 1):
                log_str = (
                    f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}] "
                    f"lr: {optimizer.param_groups[0]['lr']}, "
                )
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.3f}, "
                print(log_str.rstrip(", "))
                running_losses = {}

        scheduler.step()
        torch.save(
            model.state_dict(),
            f"vis4d-workspace/retinanet_coco_epoch_{epoch + 1}.pt",
        )
        validation_loop(model)
    print("training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    args = parser.parse_args()
    if args.ckpt is None:
        retinanet.to(device)
        training_loop(retinanet)
    else:
        retinanet.to(device)
        if args.ckpt == "mmdet":
            weights = (
                "mmdet://retinanet/retinanet_r50_fpn_2x_coco/"
                "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
            )
            load_model_checkpoint(retinanet, weights, rev_keys=REV_KEYS)
        else:
            ckpt = torch.load(args.ckpt)
            retinanet.load_state_dict(ckpt)
        validation_loop(retinanet)
