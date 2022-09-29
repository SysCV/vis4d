"""RetinaNet coco training example."""
import argparse
import copy
import warnings
from time import perf_counter
from typing import List, Optional, Tuple, Union

import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
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
from vis4d.op.detect.retinanet import (
    Dense2Det,
    RetinaNetHead,
    RetinaNetOut,
    RetinaNetLoss,
    RetinaNetLosses,
    get_default_box_matcher,
    get_default_box_sampler,
)
from vis4d.op.detect.faster_rcnn_test import identity_collate, normalize
from vis4d.op.detect.rcnn import DetOut
from vis4d.op.fpp.fpn import FPN
from vis4d.op.utils import load_model_checkpoint
from vis4d.optim.warmup import LinearLRWarmup
from vis4d.struct_to_revise import Boxes2D, InputSample

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
    ) -> Union[
        Tuple[RetinaNetLosses, RetinaNetOut], Tuple[DetOut, RetinaNetOut]
    ]:
        """Forward."""
        if target_boxes is not None:
            assert target_classes is not None
            return self._forward_train(
                images, images_hw, target_boxes, target_classes
            )
        return self._forward_test(images, images_hw)

    def _forward_train(
        self,
        images: torch.Tensor,
        images_hw: List[Tuple[int, int]],
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
    ) -> Tuple[RetinaNetLosses, RetinaNetOut]:
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
        return losses, outputs

    def _forward_test(
        self, images: torch.Tensor, images_hw: List[Tuple[int, int]]
    ) -> Tuple[DetOut, RetinaNetOut]:
        """Forward testing stage."""
        features = self.fpn(self.backbone(images))
        outs = self.retinanet_head(features[-5:])
        dets = self.transform_outs(
            class_outs=outs.cls_score,
            regression_outs=outs.bbox_pred,
            images_hw=images_hw,
        )
        return dets, outs


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
    preds = []
    class_ids_to_coco = {i: s for s, i in coco_det_map.items()}
    print("Running validation...")
    for i, data in enumerate(tqdm(test_loader)):
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

        # if i == 99:
        #     break
    _, log_str = coco_val_loader.evaluate("detect", preds, gts)
    print(log_str)


def training_loop(model):
    """Training loop."""
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

    running_losses = {}
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            for d in data[0]:
                d.images.tensor = normalize(d.images.tensor)
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
            losses, outputs = model(inputs, inputs_hw, gt_boxes, gt_class_ids)
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
                # model.visualize_proposals(inputs, outputs)
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
            from vis4d.op.detect.retinanet_test import REV_KEYS

            weights = (
                "mmdet://retinanet/retinanet_r50_fpn_2x_coco/"
                "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
            )
            load_model_checkpoint(retinanet.backbone, weights, REV_KEYS)
            load_model_checkpoint(retinanet.fpn, weights, REV_KEYS)
            load_model_checkpoint(retinanet.retinanet_head, weights, REV_KEYS)
        else:
            ckpt = torch.load(args.ckpt)
            retinanet.load_state_dict(ckpt)
        validation_loop(retinanet)
