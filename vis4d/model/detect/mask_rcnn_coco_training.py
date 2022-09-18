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

from vis4d.common_to_revise.data_pipelines import default
from vis4d.common_to_revise.datasets import coco_det_map, coco_train, coco_val
from vis4d.data.io import HDF5Backend
from vis4d.data_to_revise import (
    BaseDatasetHandler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data_to_revise.transforms import Resize
from vis4d.op.base.resnet import ResNet
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
    postprocess_dets,
)
from vis4d.op.detect.rpn import RPNLoss, RPNLosses
from vis4d.op.detect.util import apply_mask
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
        dets = self.transform_outs(
            *outs.roi, boxes=outs.proposals.boxes, images_hw=images_hw
        )
        mask_outs = self.mask_head(features[2:-1], dets.boxes)
        post_dets = DetOut(
            boxes=postprocess_dets(dets.boxes, images_hw, original_hw),
            scores=dets.scores,
            class_ids=dets.class_ids,
        )
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
coco_val_loader = coco_val()
test_sample_mapper = BaseSampleMapper(data_backend=HDF5Backend())
test_sample_mapper.setup_categories(coco_det_map)
test_transforms = [
    Resize(shape=test_resolution, keep_ratio=True, align_long_edge=True)
]
test_data = BaseDatasetHandler(
    [ScalabelDataset(coco_val_loader, False, test_sample_mapper)],
    transformations=test_transforms,
)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    collate_fn=identity_collate,
    num_workers=2,
)


@torch.no_grad()
def validation_loop(model):
    """Validate current model with test dataset."""
    model.eval()
    gts = []
    det_preds, ins_preds = [], []
    class_ids_to_coco = {i: s for s, i in coco_det_map.items()}
    print("Running validation...")
    for _, data in enumerate(tqdm(test_loader)):
        data = data[0][0]
        image = data.images.tensor.to(device)
        original_wh = (
            data.metadata[0].size.width,
            data.metadata[0].size.height,
        )
        output_wh = (image.size(3), image.size(2))

        dets, masks = mask_rcnn(
            normalize(image),
            [(output_wh[1], output_wh[0])],
            original_hw=[(original_wh[1], original_wh[0])],
        )
        boxes, scores, class_ids = dets.boxes, dets.scores, dets.class_ids

        box_pred = Boxes2D(
            torch.cat([boxes[0], scores[0].unsqueeze(-1)], -1), class_ids[0]
        )
        det_pred = copy.deepcopy(data.metadata[0])
        det_pred.labels = box_pred.to_scalabel(class_ids_to_coco)
        det_preds.append(det_pred)
        mask_pred = InstanceMasks(
            masks.masks[0], class_ids[0], score=scores[0], detections=box_pred
        )
        ins_pred = copy.deepcopy(data.metadata[0])
        ins_pred.labels = mask_pred.to_scalabel(class_ids_to_coco)
        ins_preds.append(ins_pred)

        gts.append(copy.deepcopy(data.metadata[0]))

    _, log_str = coco_val_loader.evaluate("detect", det_preds, gts)
    print(log_str)
    _, log_str = coco_val_loader.evaluate("ins_seg", ins_preds, gts)
    print(log_str)


def training_loop(model):
    """Training loop."""
    train_sample_mapper = BaseSampleMapper(
        data_backend=HDF5Backend(),
        skip_empty_samples=True,
        targets_to_load=("boxes2d", "instance_masks"),
    )
    train_sample_mapper.setup_categories(coco_det_map)
    train_transforms = default(train_resolution)

    train_data = BaseDatasetHandler(
        [ScalabelDataset(coco_train(), True, train_sample_mapper)],
        clip_bboxes_to_image=True,
        transformations=train_transforms,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=identity_collate,
        num_workers=batch_size // 2,
    )

    running_losses = {}
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = InputSample.cat(data[0], device)

            tic = perf_counter()
            inputs, inputs_hw, gt_boxes, gt_class_ids, gt_masks = (
                data.images.tensor,
                [(wh[1], wh[0]) for wh in data.images.image_sizes],
                [x.boxes for x in data.targets.boxes2d],
                [x.class_ids for x in data.targets.boxes2d],
                [x.masks for x in data.targets.instance_masks],
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
