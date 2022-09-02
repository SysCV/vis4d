"""Mask RCNN coco training example."""
import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common.data_pipelines import default
from vis4d.common.datasets import coco_det_map, coco_train, coco_val
from vis4d.common.io import HDF5Backend
from vis4d.data import BaseDatasetHandler, BaseSampleMapper, ScalabelDataset
from vis4d.data.transforms import Resize
from vis4d.op.backbone.resnet import ResNet
from vis4d.op.detect.faster_rcnn import (
    FasterRCNN,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
    get_sampled_targets,
)
from vis4d.op.detect.faster_rcnn_test import identity_collate, normalize
from vis4d.op.heads.dense_head.rpn import RPNLoss, RPNLosses
from vis4d.op.heads.roi_head.rcnn import (
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
from vis4d.op.utils import load_model_checkpoint
from vis4d.struct import Boxes2D, Detections, InputSample, InstanceMasks, Masks

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^roi_head.mask_head\.", "mask_head."),
    (r"^convs\.", "mask_head.convs."),
    (r"^upsample\.", "mask_head.upsample."),
    (r"^conv_logits\.", "mask_head.conv_logits."),
    (r"^roi_head\.", "faster_rcnn_heads.roi_head."),
    (r"^rpn_head\.", "faster_rcnn_heads.rpn_head."),
    (r"^backbone\.", "backbone.backbone.body."),
    (r"^neck.lateral_convs\.", "backbone.backbone.fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "backbone.backbone.fpn.layer_blocks."),
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
]

warnings.filterwarnings("ignore")

log_step = 100
num_epochs = 12
batch_size = 8
learning_rate = 0.02 / 16 * batch_size
train_resolution = (800, 1333)
test_resolution = (800, 1333)
device = torch.device("cuda")

training = False


class MaskRCNNModel(nn.Module):
    """Mask RCNN wrapper class for checkpointing etc."""

    def __init__(self) -> None:
        """Init."""
        super().__init__()
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        self.backbone = ResNet("resnet50", pretrained=True, trainable_layers=3)
        self.faster_rcnn_heads = FasterRCNN(
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
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_masks: Optional[List[torch.Tensor]] = None,
    ) -> Union[
        Tuple[RPNLosses, RCNNLosses, MaskRCNNLosses], Union[DetOut, MaskOut]
    ]:
        """Forward."""
        if target_boxes is not None:
            assert target_classes is not None
            return self._forward_train(
                images, target_boxes, target_classes, target_masks
            )
        return self._forward_test(images)

    def _forward_train(
        self,
        images: torch.Tensor,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
        target_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[RPNLosses, RCNNLosses, MaskRCNNLosses]:
        """Forward training stage."""
        features = self.backbone(images)
        outputs = self.faster_rcnn_heads(
            features, target_boxes, target_classes
        )
        mask_pred = self.mask_head(features[2:-1], outputs.proposals.boxes)

        rpn_losses = self.rpn_loss(
            *outputs.rpn,
            gt_boxes,
            gt_class_ids,
            inputs.shape,
        )
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.proposals.boxes,
            outputs.proposals.labels,
            outputs.proposals.target_boxes,
            outputs.proposals.target_classes,
        )
        assert outputs.proposals.target_indices is not None
        prop_masks = get_sampled_targets(
            target_masks, outputs.proposals.target_indices
        )
        mask_losses = self.mask_rcnn_loss(
            mask_pred,
            outputs.proposals.boxes,
            outputs.proposals.labels,
            prop_masks,
        )
        return rpn_losses, rcnn_losses, mask_losses

    def _forward_test(self, images: torch.Tensor) -> Union[DetOut, MaskOut]:
        """Forward testing stage."""
        features = self.backbone(images)
        outs = self.faster_rcnn_heads(features)
        mask_pred = self.mask_head(features[2:-1], outs.proposals.boxes)
        dets = self.transform_outs(
            *outs.roi,
            boxes=outs.proposals.boxes,
            images_shape=images.shape,
        )
        masks = self.det2mask(
            mask_outs=mask_pred,
            dets=dets,
            images_shape=images.shape,
        )
        return dets, masks


## setup model
mask_rcnn = MaskRCNNModel().to(device)

optimizer = optim.SGD(mask_rcnn.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[8, 11], gamma=0.1
)

## setup datasets
if training:
    train_sample_mapper = BaseSampleMapper(
        data_backend=HDF5Backend(), skip_empty_samples=True
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

coco_val_loader = coco_val()
test_sample_mapper = BaseSampleMapper(data_backend=HDF5Backend())
test_sample_mapper.setup_categories(coco_det_map)
test_transforms = [Resize(shape=test_resolution, keep_ratio=True)]
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
    """validate current model with test dataset."""
    model.eval()
    gts = []
    det_preds, ins_preds = [], []
    class_ids_to_coco = {i: s for s, i in coco_det_map.items()}
    print("Running validation...")
    for i, data in enumerate(tqdm(test_loader)):
        data = data[0][0]
        image = data.images.tensor.to(device)
        original_wh = (
            data.metadata[0].size.width,
            data.metadata[0].size.height,
        )
        output_wh = (image.size(3), image.size(2))

        dets, masks = mask_rcnn(normalize(image))
        boxes, scores, class_ids = dets.boxes, dets.scores, dets.class_ids

        box_pred = Boxes2D(
            torch.cat([boxes[0], scores[0].unsqueeze(-1)], -1),
            class_ids[0],
        )
        box_pred.postprocess(original_wh, output_wh)

        mask_pred = InstanceMasks(
            masks.masks[0], class_ids[0], score=scores[0], detections=box_pred
        )
        mask_pred.postprocess(original_wh, output_wh)

        det_pred = copy.deepcopy(data.metadata[0])
        det_pred.labels = box_pred.to_scalabel(class_ids_to_coco)
        det_preds.append(det_pred)
        ins_pred = copy.deepcopy(data.metadata[0])
        ins_pred.labels = mask_pred.to_scalabel(class_ids_to_coco)
        ins_preds.append(ins_pred)

        gts.append(copy.deepcopy(data.metadata[0]))

        if i == 10:
            break
    _, log_str = coco_val_loader.evaluate("detect", det_preds, gts)
    print(log_str)
    _, log_str = coco_val_loader.evaluate("ins_seg", ins_preds, gts)
    print(log_str)


def train_loop():
    """training loop."""
    running_losses = {}
    for epoch in range(num_epochs):
        mask_rcnn.train()
        for i, data in enumerate(train_loader):
            data = InputSample.cat(data[0], device)

            inputs, gt_boxes, gt_class_ids = (
                data.images.tensor,
                [x.boxes for x in data.targets.boxes2d],
                [x.class_ids for x in data.targets.boxes2d],
            )
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            rpn_losses, rcnn_losses = mask_rcnn(
                normalize(inputs), gt_boxes, gt_class_ids
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
                log_str = f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}] "
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.3f}, "
                print(log_str.rstrip(", "))
                running_losses = {}

        scheduler.step()
        torch.save(
            mask_rcnn.state_dict(), f"maskrcnn_coco_epoch_{epoch + 1}.pt"
        )
        validation_loop(mask_rcnn)
    print("training done.")


if __name__ == "__main__":
    # train_loop()

    weights = (
        "mmdet://mask_rcnn/mask_rcnn_r50_fpn_2x_coco/"
        "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_"
        "20200505_003907-3e542a40.pth"
    )
    load_model_checkpoint(mask_rcnn, weights, REV_KEYS)
    validation_loop(mask_rcnn)
