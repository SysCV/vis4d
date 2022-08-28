"""Faster RCNN coco training example."""
import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common.data_pipelines import default
from vis4d.common.datasets import coco_det_map, coco_train, coco_val
from vis4d.data import BaseDatasetHandler, BaseSampleMapper, ScalabelDataset
from vis4d.data.transforms import Resize
from vis4d.op.backbone.torchvision import ResNet
from vis4d.op.detect.faster_rcnn import (
    FasterRCNN,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.faster_rcnn_test import identity_collate, normalize
from vis4d.op.heads.dense_head.rpn import RPNLoss, RPNLosses
from vis4d.op.heads.roi_head.rcnn import (
    RCNNLoss,
    RCNNLosses,
    RoI2Det,
)
from vis4d.struct import Boxes2D, Detections, InputSample

log_step = 100
num_epochs = 12
learning_rate = 0.02 / 4
batch_size = 4
train_resolution = (800, 1333)
test_resolution = (800, 1333)
device = torch.device("cuda")


class FasterRCNNModel(nn.Module):
    """Faster RCNN wrapper class for checkpointing etc."""

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
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.transform_outs = RoI2Det(rcnn_bbox_encoder, score_threshold=0.05)

    def forward(
        self,
        images: torch.Tensor,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> Union[Tuple[RPNLosses, RCNNLosses], List[Detections]]:
        """Forward."""
        if target_boxes is not None:
            assert target_classes is not None
            return self._forward_train(images, target_boxes, target_classes)
        return self._forward_test(images)

    def _forward_train(
        self,
        images: torch.Tensor,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[RPNLosses, RCNNLosses]:
        """Forward training stage."""
        features = self.backbone(images)
        outputs = self.faster_rcnn_heads(
            features, target_boxes, target_classes
        )

        rpn_losses = self.rpn_loss(
            outputs.rpn_cls_out,
            outputs.rpn_reg_out,
            gt_boxes,
            gt_class_ids,
            inputs.shape,
        )
        rcnn_losses = self.rcnn_loss(
            outputs.roi_cls_out,
            outputs.roi_reg_out,
            outputs.proposal_boxes,
            outputs.proposal_labels,
            outputs.proposal_target_boxes,
            outputs.proposal_target_classes,
        )
        return rpn_losses, rcnn_losses

    def _forward_test(self, images: torch.Tensor) -> List[Detections]:
        """Forward testing stage."""
        features = self.backbone(images)
        outs = self.faster_rcnn_heads(features)
        dets = self.transform_outs(
            class_outs=outs.roi_cls_out,
            regression_outs=outs.roi_reg_out,
            boxes=outs.proposal_boxes,
            images_shape=images.shape,
        )
        return dets


## setup model
faster_rcnn = FasterRCNNModel()

optimizer = optim.SGD(faster_rcnn.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[8, 11], gamma=0.1
)

## setup datasets
train_sample_mapper = BaseSampleMapper(skip_empty_samples=True)
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
test_sample_mapper = BaseSampleMapper()
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

## validation loop
@torch.no_grad()
def validation_loop(model):
    """validate current model with test dataset"""
    model.eval()
    gts = []
    preds = []
    class_ids_to_coco = {i: s for s, i in coco_det_map.items()}
    print("Running validation...")
    for data in tqdm(test_loader):
        data = data[0][0]
        image = data.images.tensor.cuda()
        original_wh = (
            data.metadata[0].size.width,
            data.metadata[0].size.height,
        )
        output_wh = (image.size(3), image.size(2))

        dets = faster_rcnn(normalize(image))[0]
        dets = Boxes2D(
            torch.cat([dets.boxes, dets.scores.unsqueeze(-1)], -1),
            dets.class_ids,
        )
        dets.postprocess(original_wh, output_wh)

        prediction = copy.deepcopy(data.metadata[0])
        prediction.labels = dets.to_scalabel(class_ids_to_coco)

        preds.append(prediction)
        gts.append(copy.deepcopy(data.metadata[0]))
    _, log_str = coco_val_loader.evaluate("detect", preds, gts)
    print(log_str)


## training loop
running_losses = {}
faster_rcnn.to(device)
for epoch in range(num_epochs):
    faster_rcnn.train()
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
        rpn_losses, rcnn_losses = faster_rcnn(
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
    torch.save(faster_rcnn.state_dict(), f"frcnn_coco_epoch_{epoch}.pt")
    validation_loop(faster_rcnn)
print("training done.")
