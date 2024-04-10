"""Faster RCNN RPN Head."""

from __future__ import annotations

from math import prod
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms

from vis4d.common.typing import TorchLossFunc
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.box2d import bbox_clip, filter_boxes_by_area
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder, DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import Matcher, MaxIoUMatcher
from vis4d.op.box.samplers import RandomSampler, Sampler
from vis4d.op.loss.common import l1_loss

from ..layer import Conv2d
from ..typing import Proposals
from .dense_anchor import DenseAnchorHeadLoss, DenseAnchorHeadLosses


class RPNOut(NamedTuple):
    """Output of RPN head."""

    # Sigmoid input for binary classification of the anchor
    # Positive means there is an object in that anchor.
    # Each list item is for on feature pyramid level.
    cls: list[torch.Tensor]
    # 4 x number of anchors for center offets and sizes (width, height) of the
    # boxes under the anchor.
    # Each list item is for on feature pyramid level.
    box: list[torch.Tensor]


def get_default_rpn_box_codec(
    target_means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    target_stds: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> tuple[DeltaXYWHBBoxEncoder, DeltaXYWHBBoxDecoder]:
    """Get the default bounding box encoder and decoder for RPN."""
    return (
        DeltaXYWHBBoxEncoder(target_means, target_stds),
        DeltaXYWHBBoxDecoder(target_means, target_stds),
    )


class RPNHead(nn.Module):
    """Faster RCNN RPN Head.

    Creates RPN network output from a multi-scale feature map input.
    """

    rpn_conv: nn.Module

    def __init__(
        self,
        num_anchors: int,
        num_convs: int = 1,
        in_channels: int = 256,
        feat_channels: int = 256,
        start_level: int = 2,
    ) -> None:
        """Creates an instance of the class.

        Args:
            num_anchors (int): Number of anchors per cell.
            num_convs (int, optional): Number of conv layers before RPN heads.
                Defaults to 1.
            in_channels (int, optional): Feature channel size of input feature
                maps. Defaults to 256.
            feat_channels (int, optional): Feature channel size of conv layers.
                Defaults to 256.
            start_level (int, optional): starting level of feature maps.
                Defaults to 2.
        """
        super().__init__()
        self.start_level = start_level

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

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Init RPN weights."""
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, features: list[torch.Tensor]) -> RPNOut:
        """Forward pass of RPN."""
        cls_outs, box_outs = [], []
        for feat in features[self.start_level :]:
            feat = self.rpn_conv(feat)
            cls_outs += [self.rpn_cls(feat)]
            box_outs += [self.rpn_box(feat)]
        return RPNOut(cls=cls_outs, box=box_outs)

    def __call__(self, features: list[torch.Tensor]) -> RPNOut:
        """Type definition."""
        return self._call_impl(features)


class RPN2RoI(nn.Module):
    """Generate Proposals (RoIs) from RPN network output.

    This class acts as a stateless functor that does the following:
    1. Create anchor grid for feature grids (classification and regression
        outputs) at all scales.
    For each image
        For each level
            2. Get a topk pre-selection of flattened classification scores and
                box energies from feature output before NMS.
        3. Decode class scores and box energies into proposal boxes, apply NMS.
    Return proposal boxes for all images.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_decoder: None | DeltaXYWHBBoxDecoder = None,
        num_proposals_pre_nms_train: int = 2000,
        num_proposals_pre_nms_test: int = 1000,
        max_per_img: int = 1000,
        proposal_nms_threshold: float = 0.7,
        min_proposal_size: tuple[int, int] = (0, 0),
    ) -> None:
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Creates anchor grid serving as
                for bounding box regression.
            box_decoder (DeltaXYWHBBoxDecoder, optional): decodes box energies
                predicted by the network into 2D bounding box parameters.
                Defaults to None. If None, uses the default decoder.
            num_proposals_pre_nms_train (int, optional): How many boxes are
                kept prior to NMS during training. Defaults to 2000.
            num_proposals_pre_nms_test (int, optional): How many boxes are
                kept prior to NMS during inference. Defaults to 1000.
            max_per_img (int, optional): Maximum boxes per image.
                Defaults to 1000.
            proposal_nms_threshold (float, optional): NMS threshold on proposal
                boxes. Defaults to 0.7.
            min_proposal_size (tuple[int, int], optional): Minimum size of a
                proposal box. Defaults to (0, 0).
        """
        super().__init__()
        self.anchor_generator = anchor_generator

        if box_decoder is None:
            _, self.box_decoder = get_default_rpn_box_codec()
        else:
            self.box_decoder = box_decoder

        self.max_per_img = max_per_img
        self.min_proposal_size = min_proposal_size
        self.num_proposals_pre_nms_train = num_proposals_pre_nms_train
        self.num_proposals_pre_nms_test = num_proposals_pre_nms_test
        self.proposal_nms_threshold = proposal_nms_threshold

    def _get_params_per_level(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a topk pre-selection of parameters.

        The parameters include flattened classification scores and box
        energies from feature output per level per image before nms.

        Args:
            cls_out (torch.Tensor): [C, H, W] classification scores at a
                particular scale.
            reg_out (torch.Tensor): [C, H, W] regression parameters at a
                particular scale.
            anchors (torch.Tensor): [H*W, 4] anchor boxes per cell.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Topk flattened
                classification, regression outputs and corresponding anchors.
        """
        assert cls_out.size()[-2:] == reg_out.size()[-2:], (
            f"Shape mismatch: cls_out({cls_out.size()[-2:]}), reg_out("
            f"{reg_out.size()[-2:]})."
        )
        cls_out = cls_out.permute(1, 2, 0).reshape(-1).sigmoid()
        reg_out = reg_out.permute(1, 2, 0).reshape(-1, 4)
        if self.training:
            num_proposals_pre_nms = self.num_proposals_pre_nms_train
        else:
            num_proposals_pre_nms = self.num_proposals_pre_nms_test

        if 0 < num_proposals_pre_nms < cls_out.shape[0]:
            cls_out_ranked, rank_inds = cls_out.sort(descending=True)
            topk_inds = rank_inds[:num_proposals_pre_nms]
            cls_out = cls_out_ranked[:num_proposals_pre_nms]
            reg_out = reg_out[topk_inds, :]
            anchors = anchors[topk_inds, :]

        return cls_out, reg_out, anchors

    def _decode_multi_level_outputs(
        self,
        cls_out_all: list[torch.Tensor],
        reg_out_all: list[torch.Tensor],
        anchors_all: list[torch.Tensor],
        level_all: list[torch.Tensor],
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode box energies into proposals for a single image, post-process.

        Post-processing happens via NMS. NMS is performed per level.
        Afterwards, select topk proposals.

        Args:
            cls_out_all (list[torch.Tensor]): topk class scores per level.
            reg_out_all (list[torch.Tensor]): topk regression params per level.
            anchors_all (list[torch.Tensor]): topk anchor boxes per level.
            level_all (list[torch.Tensor]): tensors indicating level per entry.
            image_hw (tuple[int, int]): image size.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: decoded proposal boxes & scores.
        """
        scores = torch.cat(cls_out_all)
        levels = torch.cat(level_all)

        proposals = bbox_clip(
            self.box_decoder(torch.cat(anchors_all), torch.cat(reg_out_all)),
            image_hw,
        )

        proposals, mask = filter_boxes_by_area(
            proposals, min_area=prod(self.min_proposal_size)
        )
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
        else:  # pragma: no cover
            return proposals.new_zeros(0, 4), scores.new_zeros(0)
        return proposals, scores

    def forward(
        self,
        class_outs: list[torch.Tensor],
        regression_outs: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> Proposals:
        """Compute proposals from RPN network outputs.

        Generate anchor grid for all scales.
        For each batch element:
            Compute classification, regression and anchor pairs for all scales.
            Decode those pairs into proposals, post-process with NMS.

        Args:
            class_outs (list[torch.Tensor]): [N, 1 * A, H, W] per scale.
            regression_outs (list[torch.Tensor]): [N, 4 * A, H, W] per scale.
            images_hw (list[tuple[int, int]]): list of image sizes.

        Returns:
            Proposals: proposal boxes and scores.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = class_outs[0].device
        featmap_sizes: list[tuple[int, int]] = [
            featmap.size()[-2:] for featmap in class_outs  # type: ignore
        ]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals, scores = [], []
        for img_id, image_hw in enumerate(images_hw):
            cls_out_all, reg_out_all, anchors_all, level_all = [], [], [], []
            for level, (cls_outs, reg_outs, anchor_grid) in enumerate(
                zip(class_outs, regression_outs, anchor_grids)
            ):
                cls_out, reg_out, anchors = self._get_params_per_level(
                    cls_outs[img_id], reg_outs[img_id], anchor_grid
                )
                cls_out_all += [cls_out]
                reg_out_all += [reg_out]
                anchors_all += [anchors]
                level_all += [
                    cls_out.new_full((len(cls_out),), level, dtype=torch.long)
                ]

            box, score = self._decode_multi_level_outputs(
                cls_out_all, reg_out_all, anchors_all, level_all, image_hw
            )
            proposals.append(box)
            scores.append(score)
        return Proposals(proposals, scores)


class RPNLosses(NamedTuple):
    """RPN loss container."""

    rpn_loss_cls: torch.Tensor
    rpn_loss_bbox: torch.Tensor


class RPNLoss(DenseAnchorHeadLoss):
    """Loss of region proposal network."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        matcher: Matcher | None = None,
        sampler: Sampler | None = None,
        loss_cls: TorchLossFunc = F.binary_cross_entropy_with_logits,
        loss_bbox: TorchLossFunc = l1_loss,
    ):
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Generates anchor grid priors.
            box_encoder (DeltaXYWHBBoxEncoder): Encodes bounding boxes to the
                desired network output.
            matcher (Matcher): Matches ground truth boxes to anchor grid
                priors. Defaults to None. If None, uses MaxIoUMatcher.
            sampler (Sampler): Samples anchors for training. Defaults to None.
                If None, uses RandomSampler.
            loss_cls (TorchLossFunc): Classification loss function. Defaults to
                F.binary_cross_entropy_with_logits.
            loss_bbox (TorchLossFunc): Regression loss function. Defaults to
                l1_loss.
        """
        matcher = (
            MaxIoUMatcher(
                thresholds=[0.3, 0.7],
                labels=[0, -1, 1],
                allow_low_quality_matches=True,
                min_positive_iou=0.3,
            )
            if matcher is None
            else matcher
        )

        sampler = (
            RandomSampler(batch_size=256, positive_fraction=0.5)
            if sampler is None
            else sampler
        )

        super().__init__(
            anchor_generator,
            box_encoder,
            matcher,
            sampler,
            loss_cls,
            loss_bbox,
        )

    def forward(
        self,
        cls_outs: list[torch.Tensor],
        reg_outs: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
        target_class_ids: list[torch.Tensor | float] | None = None,
    ) -> DenseAnchorHeadLosses:
        """Compute RPN classification and regression losses.

        Args:
            cls_outs (list[torch.Tensor]): Network classification outputs
                at all scales.
            reg_outs (list[torch.Tensor]): Network regression outputs
                at all scales.
            target_boxes (list[torch.Tensor]): Target bounding boxes.
            images_hw (list[tuple[int, int]]): Image dimensions
                without padding.
            target_class_ids (list[torch.Tensor] | None): Target class labels.

        Returns:
            DenseAnchorHeadLosses: Classification and regression losses.
        """
        return super().forward(
            cls_outs, reg_outs, target_boxes, images_hw, target_class_ids
        )
