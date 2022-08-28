"""Faster RCNN tests."""
import unittest
from typing import List, NamedTuple, Optional, Tuple

import skimage
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

# TODO how to handle category IDs?
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from vis4d.common.datasets import bdd100k_track_map, bdd100k_track_sample
from vis4d.data.utils import transform_bbox
from vis4d.model.heads.dense_head.rpn import RPNLoss
from vis4d.model.heads.roi_head.rcnn import RCNNLoss, TransformRCNNOutputs
from vis4d.struct import Boxes2D
from vis4d.vis.image import imshow_bboxes

from .faster_rcnn import (
    FasterRCNN,
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)


def normalize(img: torch.Tensor) -> torch.Tensor:
    pixel_mean = (123.675, 116.28, 103.53)
    pixel_std = (58.395, 57.12, 57.375)
    pixel_mean = torch.tensor(pixel_mean, device=img.device).view(-1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=img.device).view(-1, 1, 1)
    img = (img - pixel_mean) / pixel_std
    return img


def url_to_tensor(
    url: str, im_wh: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    image = skimage.io.imread(url)
    if im_wh is not None:
        image = skimage.transform.resize(image, im_wh) * 255
    return normalize(
        torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).contiguous()
    )


class TorchResNetBackbone(nn.Module):
    """
    @fyu Leave it here for now. We will move it to a separate file later.
    """

    def __init__(
        self, name: str, pretrained: bool = True, trainable_layers: int = 3
    ):
        super().__init__()
        self.backbone = resnet_fpn_backbone(
            name, pretrained=pretrained, trainable_layers=trainable_layers
        )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(images).values())


class SampleDataset(Dataset):
    def __init__(
        self,
        return_frame_id: bool = False,
        im_wh: Optional[Tuple[int, int]] = None,
    ):
        self.return_frame_id = return_frame_id
        self.im_wh = im_wh
        self.scalabel_data = bdd100k_track_sample()

    def __len__(self):
        return len(self.scalabel_data.frames)

    def __getitem__(self, item):
        frame = self.scalabel_data.frames[item]
        img = url_to_tensor(frame.url, im_wh=self.im_wh)
        labels = Boxes2D.from_scalabel(frame.labels, bdd100k_track_map)
        if self.im_wh is not None:
            trans_mat = torch.eye(3)
            trans_mat[0, 0] = self.im_wh[0] / img.size(3)
            trans_mat[1, 1] = self.im_wh[1] / img.size(2)
            labels.boxes[:, :4] = transform_bbox(
                trans_mat, labels.boxes[:, :4]
            )
        if self.return_frame_id:
            return img, labels.boxes, labels.class_ids, frame.frameIndex - 165
        else:
            return img, labels.boxes, labels.class_ids


def identity_collate(batch):
    return tuple(zip(*batch))


TOPK_PROPOSAL_BOXES = torch.tensor(
    [
        [142.0056, 251.8184, 489.2325, 425.7317],
        [45.3251, 239.5631, 512.0000, 512.0000],
        [118.2012, 289.3944, 502.2325, 456.9539],
        [91.2816, 0.0000, 203.5461, 340.9729],
        [89.2667, 337.7622, 171.3395, 380.5127],
        [390.6346, 196.6292, 464.6955, 291.1225],
        [170.0071, 258.1156, 491.6874, 488.2177],
        [65.6316, 318.7266, 512.0000, 500.5325],
        [275.7870, 210.0989, 331.9042, 258.9714],
        [26.8909, 16.0752, 512.0000, 512.0000],
        [0.0000, 189.0641, 512.0000, 477.8121],
        [171.5342, 216.2149, 480.8415, 412.2690],
        [450.5605, 279.5934, 512.0000, 510.4569],
        [93.5586, 30.9820, 198.4854, 344.2863],
        [99.1633, 127.1192, 512.0000, 473.8350],
        [388.4907, 184.0709, 467.7780, 300.4931],
        [27.4126, 356.1322, 498.5256, 484.0288],
        [223.7903, 235.3380, 511.3469, 428.4519],
        [98.5912, 0.0000, 248.7077, 365.3753],
        [238.8812, 203.0925, 275.6957, 264.5228],
        [237.5958, 281.2570, 506.6336, 512.0000],
        [405.0393, 316.2453, 461.7299, 365.4779],
        [44.7720, 330.9578, 208.6164, 405.6234],
        [88.0496, 73.8804, 182.5938, 351.1889],
        [63.3251, 336.8067, 110.4334, 353.2487],
        [47.6086, 0.0000, 186.6715, 369.1220],
        [120.5068, 70.5646, 235.4800, 383.8726],
        [144.7434, 163.4856, 473.4056, 434.1283],
        [206.4816, 264.3892, 470.8024, 381.8293],
        [0.0000, 0.0000, 213.2711, 512.0000],
        [63.8757, 200.4689, 512.0000, 428.2086],
        [60.1210, 333.9038, 258.4829, 398.9240],
        [77.4594, 52.2637, 205.1834, 419.5437],
        [10.0437, 0.0000, 332.4379, 512.0000],
        [436.1147, 274.8861, 462.0505, 294.1747],
        [254.1903, 207.6381, 334.4627, 258.4989],
        [171.3309, 313.9091, 457.9484, 408.7146],
        [168.0634, 286.0563, 450.9991, 400.1044],
        [0.0000, 380.8892, 491.8259, 501.7971],
        [157.3955, 347.2469, 190.1704, 397.3947],
        [281.7195, 334.5407, 316.5000, 419.0572],
        [162.4470, 339.1355, 431.7346, 419.9622],
        [279.0762, 316.3195, 320.0567, 409.0187],
        [122.0617, 27.5344, 206.6314, 348.1996],
        [331.1177, 0.0000, 512.0000, 72.8454],
        [93.6474, 337.5839, 212.9738, 383.8267],
        [159.2087, 351.2808, 198.7565, 394.4629],
        [94.0561, 335.9900, 188.0700, 376.9784],
        [0.0000, 0.0000, 126.5411, 512.0000],
        [96.8320, 86.4032, 255.4249, 485.9869],
        [60.9948, 339.5186, 103.1848, 356.7444],
        [152.5533, 263.9601, 411.8368, 385.4301],
        [146.5829, 282.5284, 431.5514, 512.0000],
        [400.0861, 300.4162, 465.2494, 363.3577],
        [46.1462, 355.4701, 174.2954, 399.0386],
        [63.4404, 337.7604, 98.0852, 359.3974],
        [278.3034, 203.6742, 327.3413, 267.3003],
        [285.7585, 336.4688, 315.5070, 410.4036],
        [135.0184, 358.8753, 423.9805, 430.1019],
        [136.7889, 236.4654, 398.0069, 444.4665],
        [155.1931, 255.4130, 358.8625, 404.3157],
        [79.3220, 31.5286, 290.7912, 350.5450],
        [58.8012, 335.6199, 101.0351, 352.4113],
        [43.7260, 366.3886, 187.4183, 404.1813],
        [266.6708, 250.7574, 324.8056, 307.1678],
        [32.9305, 0.0000, 181.0987, 512.0000],
        [156.4861, 345.0082, 198.2086, 394.3762],
        [37.7532, 324.2015, 154.9769, 421.2256],
        [55.2943, 336.0358, 125.6785, 351.6338],
        [57.7155, 344.4187, 295.6890, 413.3869],
        [33.4714, 405.1226, 512.0000, 512.0000],
        [63.4223, 335.2281, 163.2025, 386.3951],
        [392.0016, 186.6851, 463.1364, 267.9714],
        [428.1172, 252.3178, 507.8842, 503.4626],
        [211.9772, 247.2469, 457.5269, 346.3222],
        [134.5331, 91.8276, 324.0479, 487.4446],
        [178.7352, 269.4692, 381.3005, 415.9282],
        [44.2539, 375.0102, 194.5286, 407.8476],
        [57.6669, 343.9175, 164.1898, 394.3400],
        [196.1217, 21.7895, 512.0000, 512.0000],
        [227.4097, 215.4822, 512.0000, 385.8077],
        [388.1335, 221.6628, 464.4626, 281.7422],
        [358.6649, 188.1329, 472.3539, 306.7631],
        [102.0133, 354.4927, 162.2373, 379.4893],
        [235.0134, 203.4016, 336.5229, 265.1878],
        [62.3884, 331.8939, 114.1846, 351.3175],
        [379.9470, 319.8063, 456.8282, 368.2285],
        [159.3253, 155.8603, 369.6798, 477.2817],
        [228.2653, 252.9330, 250.9940, 303.8557],
        [61.1279, 327.6759, 111.1346, 348.2917],
        [88.7114, 31.7487, 161.3475, 351.1284],
        [0.0000, 45.3947, 69.0168, 511.7769],
        [197.0290, 299.4307, 512.0000, 470.0799],
        [443.1889, 307.5229, 478.5656, 512.0000],
        [68.6224, 353.0646, 175.5813, 394.3654],
        [62.9085, 343.1974, 96.3924, 364.0455],
        [280.4498, 333.8660, 304.9383, 412.5363],
        [318.1721, 250.3777, 503.1625, 512.0000],
        [0.0000, 308.6399, 512.0000, 468.3215],
        [45.9215, 329.6155, 113.9582, 392.6399],
    ]
)

DET0_BOXES = torch.tensor(
    [
        [147.7766, 248.9840, 488.3863, 424.8228],
        [88.7897, 11.5224, 206.9523, 343.1987],
        [93.1193, 265.4994, 501.5325, 504.1375],
    ]
)

DET0_SCORES = torch.tensor([0.9911, 0.9884, 0.8300])
DET0_CLASS_IDS = torch.tensor([15, 75, 57])

DET1_BOXES = torch.tensor(
    [
        [217.3183, 277.0768, 426.0976, 461.2431],
        [292.5947, 320.2501, 511.1806, 505.7900],
        [331.9866, 259.2177, 378.7979, 289.9914],
        [106.6400, 267.3953, 116.5251, 293.8373],
        [404.5693, 226.8489, 435.8844, 258.8338],
        [102.8251, 303.2993, 114.8895, 330.4320],
        [66.6481, 240.0991, 133.2655, 265.8181],
        [240.2145, 177.2466, 250.2043, 212.2566],
        [63.6218, 263.7009, 130.0857, 305.0337],
        [203.3486, 255.5775, 271.6653, 355.6212],
        [417.3006, 285.7568, 455.4529, 350.9530],
        [72.4229, 298.9252, 128.9032, 335.3980],
        [456.4732, 262.9994, 493.9896, 333.4376],
    ]
)

DET1_SCORES = torch.tensor(
    [
        0.8112,
        0.7819,
        0.7430,
        0.7327,
        0.7229,
        0.6573,
        0.6535,
        0.6411,
        0.6383,
        0.5951,
        0.5717,
        0.5571,
        0.5462,
    ]
)

DET1_CLASS_IDS = torch.tensor(
    [57, 57, 56, 73, 58, 73, 73, 39, 73, 56, 58, 73, 58]
)


class FasterRCNNTest(unittest.TestCase):
    """Faster RCNN test class."""

    def test_inference(self):
        image1 = url_to_tensor(
            "https://farm1.staticflickr.com/106/311161252_33d75830fd_z.jpg",
            (512, 512),
        )
        image2 = url_to_tensor(
            "https://farm4.staticflickr.com/3217/2980271186_9ec726e0fa_z.jpg",
            (512, 512),
        )
        sample_images = torch.cat([image1, image2])

        faster_rcnn = FasterRCNN(
            backbone=TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
            num_classes=80,
            weights=(
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/"
                "faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_"
                "20200504_210434-a5d8aa15.pth"
            ),
        )
        faster_rcnn.eval()
        with torch.no_grad():
            outs = faster_rcnn(sample_images)
            transform_outs = TransformRCNNOutputs(
                faster_rcnn.rcnn_box_encoder, score_threshold=0.5
            )
            dets = transform_outs(
                class_outs=outs.roi_cls_out,
                regression_outs=outs.roi_reg_out,
                boxes=outs.proposal_boxes,
                images_shape=sample_images.shape,
            )

        _, topk = torch.topk(outs.proposal_scores[0], 100)
        assert outs.proposal_boxes[0][topk].shape[0] == 100
        assert outs.proposal_boxes[0][topk].shape.numel() == 400
        assert (
            torch.isclose(outs.proposal_boxes[0][topk], TOPK_PROPOSAL_BOXES)
            .all()
            .item()
        )
        assert torch.isclose(dets[0].boxes, DET0_BOXES).all().item()
        assert (
            torch.isclose(dets[0].scores, DET0_SCORES, atol=1e-4).all().item()
        )
        assert torch.equal(dets[0].class_ids, DET0_CLASS_IDS)
        assert torch.isclose(dets[1].boxes, DET1_BOXES).all().item()
        assert (
            torch.isclose(dets[1].scores, DET1_SCORES, atol=1e-4).all().item()
        )
        assert torch.equal(dets[1].class_ids, DET1_CLASS_IDS)

        # imshow_bboxes(image1[0], *dets[0])
        # imshow_bboxes(image2[0], *dets[1])

    def test_train(self):
        # TODO should bn be frozen during training?
        anchor_gen = get_default_anchor_generator()
        rpn_bbox_encoder = get_default_rpn_box_encoder()
        rcnn_bbox_encoder = get_default_rcnn_box_encoder()
        faster_rcnn = FasterRCNN(
            TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
            anchor_generator=anchor_gen,
            rpn_box_encoder=rpn_bbox_encoder,
            rcnn_box_encoder=rcnn_bbox_encoder,
            num_classes=8,
        )
        rpn_loss = RPNLoss(anchor_gen, rpn_bbox_encoder)
        rcnn_loss = RCNNLoss(rcnn_bbox_encoder, num_classes=8)

        optimizer = optim.SGD(faster_rcnn.parameters(), lr=0.001, momentum=0.9)

        train_data = SampleDataset()
        train_loader = DataLoader(
            train_data, batch_size=2, shuffle=True, collate_fn=identity_collate
        )

        running_losses = {}
        faster_rcnn.train()
        log_step = 1
        for epoch in range(2):
            for i, data in enumerate(train_loader):
                inputs, gt_boxes, gt_class_ids = data
                inputs = torch.cat(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = faster_rcnn(inputs, gt_boxes, gt_class_ids)
                rpn_losses = rpn_loss(
                    outputs.rpn_cls_out,
                    outputs.rpn_reg_out,
                    gt_boxes,
                    gt_class_ids,
                    inputs.shape,
                )
                rcnn_losses = rcnn_loss(
                    outputs.roi_cls_out,
                    outputs.roi_reg_out,
                    outputs.proposal_boxes,
                    outputs.proposal_labels,
                    outputs.proposal_target_boxes,
                    outputs.proposal_target_classes,
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
                    log_str = f"[{epoch + 1}, {i + 1:5d}] "
                    for k, v in running_losses.items():
                        log_str += f"{k}: {v / log_step:.3f}, "
                    print(log_str.rstrip(", "))
                    running_losses = {}

    def test_torchscript(self):
        sample_images = torch.rand((2, 3, 512, 512))
        faster_rcnn = FasterRCNN(
            backbone=TorchResNetBackbone(
                "resnet50", pretrained=True, trainable_layers=3
            ),
            weights=(
                "mmdet://faster_rcnn/faster_rcnn_r50_fpn_2x_coco/"
                "faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_"
                "20200504_210434-a5d8aa15.pth"
            ),
        )
        frcnn_scripted = torch.jit.script(faster_rcnn)
        frcnn_scripted(sample_images)
