"""Mask RCNN coco training example."""
import argparse
import copy
import warnings
from time import perf_counter
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common_to_revise.data_pipelines import default_test, default_train
from vis4d.common_to_revise.datasets import coco_train, coco_val
from vis4d.data.datasets.base import DataKeys
from vis4d.data.datasets.coco import COCO, coco_det_map
from vis4d.data.io import HDF5Backend
from vis4d.data_to_revise import (
    BaseDatasetHandler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.eval.coco import COCOEvaluator
from vis4d.op.base.resnet import ResNet
from vis4d.op.box.util import apply_mask, bbox_postprocess
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.faster_rcnn_test import identity_collate, normalize
from vis4d.op.detect.rcnn import (
    Det2Mask,
    DetOut,
    MaskOut,
    MaskRCNNHead,
    MaskRCNNLoss,
    MaskRCNNLosses,
    RCNNLoss,
    RCNNLosses,
    RoI2Det,
)
from vis4d.op.detect.rpn import RPNLoss, RPNLosses
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim.warmup import LinearLRWarmup
from vis4d.struct_to_revise import Boxes2D, InputSample, InstanceMasks

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^roi_head.mask_head\.", "mask_head."),
    (r"^convs\.", "mask_head.convs."),
    (r"^upsample\.", "mask_head.upsample."),
    (r"^conv_logits\.", "mask_head.conv_logits."),
    (r"^roi_head\.", "faster_rcnn_heads.roi_head."),
    (r"^rpn_head\.", "faster_rcnn_heads.rpn_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]

warnings.filterwarnings("ignore")

log_step = 100
num_epochs = 12
batch_size = 2  # 8
learning_rate = 0.02 / 16 * batch_size
train_resolution = (800, 1333)
test_resolution = (800, 1333)
device = torch.device("cuda")


class MaskRCNNModel(nn.Module):
    """Mask RCNN wrapper class for checkpointing etc."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.fpn = FPN(self.backbone.out_channels[2:], 256)
        self.faster_rcnn_heads = FasterRCNNHead(
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
        )
        self.mask_head = MaskRCNNHead()
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.mask_rcnn_loss = MaskRCNNLoss()
        self.transform_outs = RoI2Det(rcnn_bbox_encoder)
        self.det2mask = Det2Mask()

    def forward(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_masks: Optional[List[torch.Tensor]] = None,
        original_hw: Optional[List[Tuple[int, int]]] = None,
    ) -> Union[
        Tuple[RPNLosses, RCNNLosses, MaskRCNNLosses], Union[DetOut, MaskOut]
    ]:
        """Forward."""
        if target_boxes is not None:
            assert target_classes is not None
            return self._forward_train(
                images, images_hw, target_boxes, target_classes, target_masks
            )
        assert original_hw is not None
        return self._forward_test(images, images_hw, original_hw)

    def _forward_train(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[RPNLosses, RCNNLosses, MaskRCNNLosses]:
        """Forward training stage."""
        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )
        mask_outs = self.mask_head(
            features[2:-1], outputs.sampled_proposals.boxes
        )

        rpn_losses = self.rpn_loss(*outputs.rpn, target_boxes, images_hw)
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )
        assert outputs.sampled_target_indices is not None
        sampled_masks = apply_mask(
            outputs.sampled_target_indices, target_masks
        )[0]
        mask_losses = self.mask_rcnn_loss(
            mask_outs.mask_pred,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.classes,
            sampled_masks,
        )
        return rpn_losses, rcnn_losses, mask_losses

    def _forward_test(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        original_hw: List[Tuple[int, int]],
    ) -> Tuple[DetOut, MaskOut]:
        """Forward testing stage."""
        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        dets, scores, class_ids = self.transform_outs(
            *outs.roi, boxes=outs.proposals.boxes, images_hw=images_hw
        )
        mask_outs = self.mask_head(features[2:-1], dets)
        for i, boxs in enumerate(dets):
            dets[i] = bbox_postprocess(boxs, original_hw[i], images_hw[i])
        post_dets = DetOut(boxes=dets, scores=scores, class_ids=class_ids)
        masks = self.det2mask(
            mask_outs=mask_outs.mask_pred.sigmoid(),
            dets=post_dets,
            images_hw=original_hw,
        )
        return post_dets, masks


## setup model
mask_rcnn = MaskRCNNModel()

optimizer = optim.SGD(
    mask_rcnn.parameters(),
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

        dets, masks = mask_rcnn(images, images_hw, original_hw=original_hw)
        boxes, scores, class_ids = dets.boxes, dets.scores, dets.class_ids

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
            inputs, inputs_hw, gt_boxes, gt_class_ids, gt_masks = (
                data[DataKeys.images].to(device),
                data[DataKeys.metadata]["input_hw"],
                [b.to(device) for b in data[DataKeys.boxes2d]],
                [b.to(device) for b in data[DataKeys.boxes2d_classes]],
                [m.to(device) for m in data[DataKeys.masks]],
            )
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            rpn_losses, rcnn_losses, mask_losses = model(
                normalize(inputs), inputs_hw, gt_boxes, gt_class_ids, gt_masks
            )
            total_loss = sum((*rpn_losses, *rcnn_losses, *mask_losses))
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
            losses = dict(
                time=toc - tic,
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
                log_str = f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}] "
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.3f}, "
                print(log_str.rstrip(", "))
                running_losses = {}

        scheduler.step()
        torch.save(
            model.state_dict(),
            f"vis4d-workspace/test/maskrcnn_coco_epoch_{epoch + 1}.pt",
        )
        validation_loop(model)
    print("training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    parser.add_argument("-n", "--num_gpus", default=1, help="number of gpus")
    args = parser.parse_args()
    mask_rcnn.to(device)
    if args.ckpt is None:
        if args.num_gpus > 1:
            mask_rcnn = nn.DataParallel(
                mask_rcnn, device_ids=[device, torch.device("cuda:5")]
            )
        training_loop(mask_rcnn)
    else:
        if args.ckpt == "mmdet":
            weights = (
                "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
                "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
                "20200505_003907-3e542a40.pth"
            )
            load_model_checkpoint(mask_rcnn, weights, REV_KEYS)
        else:
            ckpt = torch.load(args.ckpt)
            mask_rcnn.load_state_dict(ckpt)
        validation_loop(mask_rcnn)
