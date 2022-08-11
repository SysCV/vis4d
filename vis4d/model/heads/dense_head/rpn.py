"""Faster RCNN RPN Head."""
from math import prod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms

from vis4d.common.bbox.anchor_generator import AnchorGenerator
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxCoder
from vis4d.common.layers import Conv2d
from vis4d.model.utils import _parse_losses, get_img_metas, load_config
from vis4d.struct import Boxes2D, LossesType, NamedTensors

from .base import (
    BaseDenseBox2DHead,
    BaseDenseBox2DHeadLoss,
    TransformDenseHeadOutputs,
)


class RPNHead(BaseDenseBox2DHead):
    """Faster RCNN RPN Head."""

    def __init__(
        self,
        num_convs: int = 1,
        in_channels: int = 256,
        feat_channels: int = 256,
        num_anchors: int = 3,
    ) -> None:
        """Init."""
        super().__init__()
        # TODO align num_anchors with anchor generator
        if num_convs > 1:
            rpn_convs = []
            for i in range(num_convs):
                if i > 0:
                    in_channels = feat_channels
                rpn_convs.append(
                    Conv2d(
                        in_channels,
                        feat_channels,
                        kernel_size=3,
                        padding=1,
                        activation=nn.ReLU(inplace=False),
                    )
                )
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = Conv2d(
                in_channels,
                feat_channels,
                kernel_size=3,
                padding=1,
                activation=nn.ReLU(inplace=True),
            )
        self.rpn_cls = Conv2d(feat_channels, num_anchors, 1)
        self.rpn_box = Conv2d(feat_channels, num_anchors * 4, 1)

        # TODO weight init

    def forward(
        self,
        features: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],]:
        """Forward pass during training stage."""
        cls_outs, box_outs = [], []
        for feat in features:
            feat = self.rpn_conv(feat)
            cls_outs += [self.rpn_cls(feat)]
            box_outs += [self.rpn_box(feat)]
        return cls_outs, box_outs


class TransformRPNOutputs(TransformDenseHeadOutputs):
    def __init__(
        self,
        num_proposals_pre_nms: int = 2000,
        max_per_img: int = 1000,
        proposal_nms_threshold: float = 0.7,
        min_proposal_size: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.max_per_img = max_per_img
        self.min_proposal_size = min_proposal_size
        self.num_proposals_pre_nms = num_proposals_pre_nms
        self.proposal_nms_threshold = proposal_nms_threshold

        self.anchor_generator = AnchorGenerator(
            scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
        )

        self.bbox_coder = DeltaXYWHBBoxCoder(
            target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)
        )

    def _get_params_per_level(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        anchors: torch.Tensor,
    ):
        assert cls_out.size()[-2:] == reg_out.size()[-2:], (
            f"Shape mismatch: cls_out({cls_out.size()[-2:]}), reg_out("
            f"{reg_out.size()[-2:]})."
        )
        cls_out = cls_out.permute(1, 2, 0).reshape(-1).sigmoid()
        reg_out = reg_out.permute(1, 2, 0).reshape(-1, 4)
        if 0 < self.num_proposals_pre_nms < cls_out.shape[0]:
            cls_out_ranked, rank_inds = cls_out.sort(descending=True)
            topk_inds = rank_inds[: self.num_proposals_pre_nms]
            cls_out = cls_out_ranked[: self.num_proposals_pre_nms]
            reg_out = reg_out[topk_inds, :]
            anchors = anchors[topk_inds, :]

        return cls_out, reg_out, anchors

    def _decode_multi_level_outputs(
        self, cls_out_all, reg_out_all, anchors_all, level_all
    ) -> Boxes2D:
        scores = torch.cat(cls_out_all)
        levels = torch.cat(level_all)
        proposals = self.bbox_coder.decode(
            torch.cat(anchors_all),
            torch.cat(reg_out_all),
            max_shape=(512, 512),
        )  # TODO replace max_shape

        from vis4d.struct.labels.boxes import filter_boxes

        proposals, mask = filter_boxes(
            proposals, min_area=prod(self.min_proposal_size)
        )  # TODO area doesnt constrain size
        scores = scores[mask]
        levels = levels[mask]

        if proposals.numel() > 0:
            keep = batched_nms(
                proposals,
                scores,
                levels,
                iou_threshold=self.proposal_nms_threshold,
            )[: self.max_per_img]
            proposals = proposals[keep]
            scores = scores[keep]
        else:
            return Boxes2D(proposals.new_zeros(0, 5))
        return Boxes2D(torch.cat([proposals, scores.unsqueeze(-1)], -1))

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """

        Args:
            class_outs (N, 1 * A, H, W)
        Returns:
            boxes
            scores
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = class_outs[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals_all = []
        for img_id in range(images_shape[0]):
            cls_out_all, reg_out_all, anchors_all, level_all = [], [], [], []
            for level in range(len(class_outs)):
                cls_out, reg_out, anchors = self._get_params_per_level(
                    class_outs[level][img_id],
                    regression_outs[level][img_id],
                    anchor_grids[level],
                )
                cls_out_all += [cls_out]
                reg_out_all += [reg_out]
                anchors_all += [anchors]
                level_all += [
                    cls_out.new_full((len(cls_out),), level, dtype=torch.long)
                ]

            proposals_all += [
                self._decode_multi_level_outputs(
                    cls_out_all, reg_out_all, anchors_all, level_all
                )
            ]
        return proposals_all


class MMDetDenseHeadLoss(BaseDenseBox2DHeadLoss):
    def __init__(self):
        super().__init__()

    def loss(
        self,
        class_outs: NamedTensors,
        regression_outs: NamedTensors,
        target_boxes: List[torch.Tensor],
        target_classes: List[torch.Tensor],
        images_shape: Tuple[int, int, int, int],
    ) -> LossesType:
        """MMDet head loss wrapper.

        Args:
            outputs: Network outputs.
            targets (List[Boxes2D]): Target 2D boxes.
            metadata (Dict): Dictionary of metadata needed for loss, e.g.
                image size, feature map strides, etc.
        Returns:
            LossesType: Dictionary of scalar loss tensors.
        """
        img_metas = get_img_metas(images_shape)
        return self.mm_dense_head.loss(
            class_outs,
            regression_outs,
            target_boxes,
            target_classes,
            img_metas,
        )
