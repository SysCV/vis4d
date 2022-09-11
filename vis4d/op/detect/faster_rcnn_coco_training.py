"""Faster RCNN coco training example."""
import argparse
import copy
import warnings
from time import perf_counter
from typing import List, Optional, Tuple, Union

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vis4d.common_to_clean.data_pipelines import default
from vis4d.common_to_clean.datasets import coco_det_map, coco_train, coco_val
from vis4d.data_to_clean import (
    BaseDatasetHandler,
    BaseSampleMapper,
    ScalabelDataset,
)
from vis4d.data_to_clean.transforms import Resize
from vis4d.op.base.resnet import ResNet
from vis4d.op.detect.faster_rcnn import (
    FasterRCNNHead,
    FRCNNOut,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.op.detect.faster_rcnn_test import identity_collate, normalize
from vis4d.op.detect.rcnn import DetOut, RCNNLoss, RCNNLosses, RoI2Det
from vis4d.op.detect.rpn import RPNLoss, RPNLosses
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.struct import Boxes2D, InputSample

warnings.filterwarnings("ignore")

log_step = 100
num_epochs = 12
batch_size = 8
learning_rate = 0.02 / 16 * batch_size
train_resolution = (800, 1333)
test_resolution = (800, 1333)
device = torch.device("cuda:4")


class FasterRCNN(nn.Module):
    """Faster RCNN wrapper class for checkpointing etc."""

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
        self.rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        self.rcnn_loss = RCNNLoss(rcnn_bbox_encoder)
        self.transform_outs = RoI2Det(rcnn_bbox_encoder)

    def forward(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> Union[
        Tuple[RPNLosses, RCNNLosses, FRCNNOut], Tuple[DetOut, FRCNNOut]
    ]:
        """Forward."""
        if target_boxes is not None:
            assert target_classes is not None
            return self._forward_train(
                images, images_hw, target_boxes, target_classes
            )
        return self._forward_test(images, images_hw)

    def visualize_proposals(
        self, images: torch.Tensor, outs: FRCNNOut, topk: int = 100
    ) -> None:
        """Visualize topk proposals."""
        from vis4d.vis.image import imshow_bboxes

        for im, boxes, scores in zip(images, *outs.proposals):
            _, topk_indices = torch.topk(scores, topk)
            imshow_bboxes(im, boxes[topk_indices])

    def _forward_train(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> Tuple[RPNLosses, RCNNLosses, FRCNNOut]:
        """Forward training stage."""
        features = self.fpn(self.backbone(images))
        outputs = self.faster_rcnn_heads(
            features, images_hw, target_boxes, target_classes
        )

        rpn_losses = self.rpn_loss(*outputs.rpn, target_boxes, images_hw)
        rcnn_losses = self.rcnn_loss(
            *outputs.roi,
            outputs.sampled_proposals.boxes,
            outputs.sampled_targets.labels,
            outputs.sampled_targets.boxes,
            outputs.sampled_targets.classes,
        )
        return rpn_losses, rcnn_losses, outputs

    def _forward_test(
        self, images: torch.Tensor, images_hw: List[Tuple[int, int]]
    ) -> Tuple[DetOut, FRCNNOut]:
        """Forward testing stage."""
        features = self.fpn(self.backbone(images))
        outs = self.faster_rcnn_heads(features, images_hw)
        dets = self.transform_outs(*outs.roi, outs.proposals.boxes, images_hw)
        return dets, outs


## setup model
faster_rcnn = FasterRCNN()

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

## validation loop
@torch.no_grad()
def validation_loop(model):
    """Validate current model with test dataset."""
    model.eval()
    gts = []
    preds = []
    class_ids_to_coco = {i: s for s, i in coco_det_map.items()}
    print("Running validation...")
    for data in tqdm(test_loader):
        data = data[0][0]
        image = data.images.tensor.to(device)
        original_wh = (
            data.metadata[0].size.width,
            data.metadata[0].size.height,
        )
        output_wh = data.images.image_sizes[0]

        (boxes, scores, class_ids), outputs = model(
            normalize(image), [(output_wh[1], output_wh[0])]
        )

        # model.visualize_proposals(image, outputs, topk=10)
        # postprocess for eval
        dets = Boxes2D(
            torch.cat([boxes[0], scores[0].unsqueeze(-1)], -1),
            class_ids[0],
        )
        dets.postprocess(original_wh, output_wh)

        prediction = copy.deepcopy(data.metadata[0])
        prediction.labels = dets.to_scalabel(class_ids_to_coco)

        preds.append(prediction)
        gts.append(copy.deepcopy(data.metadata[0]))

    _, log_str = coco_val_loader.evaluate("detect", preds, gts)
    print(log_str)


## training loop
def training_loop(model):
    """Training loop."""
    running_losses = {}
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            data = InputSample.cat(data[0], device)

            tic = perf_counter()
            inputs, inputs_hw, gt_boxes, gt_class_ids = (
                data.images.tensor,
                [(wh[1], wh[0]) for wh in data.images.image_sizes],
                [x.boxes for x in data.targets.boxes2d],
                [x.class_ids for x in data.targets.boxes2d],
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            rpn_losses, rcnn_losses, outputs = model(
                normalize(inputs), inputs_hw, gt_boxes, gt_class_ids
            )
            total_loss = sum((*rpn_losses, *rcnn_losses))
            total_loss.backward()
            optimizer.step()
            toc = perf_counter()

            # print statistics
            losses = dict(
                time=toc - tic,
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
                # model.visualize_proposals(inputs, outputs)
                log_str = f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}] "
                for k, v in running_losses.items():
                    log_str += f"{k}: {v / log_step:.3f}, "
                print(log_str.rstrip(", "))
                running_losses = {}

        scheduler.step()
        torch.save(
            model.state_dict(),
            f"vis4d-workspace/frcnn_coco_epoch_{epoch + 1}.pt",
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
        faster_rcnn.to(device)
        training_loop(faster_rcnn)
    else:
        faster_rcnn.to(device)
        if args.ckpt == "mmdet":
            from vis4d.op.detect.faster_rcnn_test import REV_KEYS

            weights = "mmdet://faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
            load_model_checkpoint(faster_rcnn.backbone, weights, REV_KEYS)
            load_model_checkpoint(faster_rcnn.fpn, weights, REV_KEYS)
            load_model_checkpoint(
                faster_rcnn.faster_rcnn_heads, weights, REV_KEYS
            )
        else:
            ckpt = torch.load(args.ckpt)
            faster_rcnn.load_state_dict(ckpt)
        validation_loop(faster_rcnn)
