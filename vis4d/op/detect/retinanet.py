"""RetinaNet."""

from __future__ import annotations

from math import prod
from typing import NamedTuple

import torch
from torch import nn
from torchvision.ops import batched_nms, sigmoid_focal_loss

from vis4d.common.typing import TorchLossFunc
from vis4d.op.box.anchor import AnchorGenerator
from vis4d.op.box.box2d import bbox_clip, filter_boxes_by_area
from vis4d.op.box.encoder import DeltaXYWHBBoxDecoder, DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import Matcher, MaxIoUMatcher
from vis4d.op.box.samplers import PseudoSampler, Sampler
from vis4d.op.loss.common import l1_loss

from .common import DetOut
from .dense_anchor import DenseAnchorHeadLoss


class RetinaNetOut(NamedTuple):
    """RetinaNet head outputs."""

    # Logits for box classification for each feature level. The logit
    # dimention is [batch_size, number of anchors * number of classes, height,
    # width].
    cls_score: list[torch.Tensor]
    # Each box has regression for all classes for each feature level. So the
    # tensor dimension is [batch_size, number of anchors * 4, height, width].
    bbox_pred: list[torch.Tensor]


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128],
    )


def get_default_box_codec() -> (
    tuple[DeltaXYWHBBoxEncoder, DeltaXYWHBBoxDecoder]
):
    """Get the default bounding box encoder."""
    return (
        DeltaXYWHBBoxEncoder(
            target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)
        ),
        DeltaXYWHBBoxDecoder(
            target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)
        ),
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default bounding box matcher."""
    return MaxIoUMatcher(
        thresholds=[0.4, 0.5],
        labels=[0, -1, 1],
        allow_low_quality_matches=True,
    )


def get_default_box_sampler() -> PseudoSampler:
    """Get default bounding box sampler."""
    return PseudoSampler()


class RetinaNetHead(nn.Module):  # TODO: Refactor to use the new API
    """RetinaNet Head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_sigmoid_cls: bool = True,
        anchor_generator: AnchorGenerator | None = None,
        box_decoder: DeltaXYWHBBoxDecoder | None = None,
        box_matcher: Matcher | None = None,
        box_sampler: Sampler | None = None,
    ):
        """Creates an instance of the class."""
        super().__init__()
        self.anchor_generator = (
            anchor_generator
            if anchor_generator is not None
            else get_default_anchor_generator()
        )
        if box_decoder is None:
            _, self.box_decoder = get_default_box_codec()
        else:
            self.box_decoder = box_decoder
        self.box_matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
        )
        self.box_sampler = (
            box_sampler
            if box_sampler is not None
            else get_default_box_sampler()
        )
        num_base_priors = self.anchor_generator.num_base_priors[0]

        if use_sigmoid_cls:
            cls_out_channels = num_classes
        else:
            cls_out_channels = num_classes + 1
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1),
            )
            self.reg_convs.append(
                nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1),
            )
        self.retina_cls = nn.Conv2d(
            feat_channels, num_base_priors * cls_out_channels, 3, padding=1
        )
        self.retina_reg = nn.Conv2d(
            feat_channels, num_base_priors * 4, 3, padding=1
        )

    def forward(self, features: list[torch.Tensor]) -> RetinaNetOut:
        """Forward pass of RetinaNet.

        Args:
            features (list[torch.Tensor]): Feature pyramid

        Returns:
            RetinaNetOut: classification score and box prediction.
        """
        cls_scores, bbox_preds = [], []
        for feat in features:
            cls_feat = feat
            reg_feat = feat
            for cls_conv in self.cls_convs:
                cls_feat = self.relu(cls_conv(cls_feat))
            for reg_conv in self.reg_convs:
                reg_feat = self.relu(reg_conv(reg_feat))
            cls_scores.append(self.retina_cls(cls_feat))
            bbox_preds.append(self.retina_reg(reg_feat))
        return RetinaNetOut(cls_score=cls_scores, bbox_pred=bbox_preds)

    def __call__(self, features: list[torch.Tensor]) -> RetinaNetOut:
        """Type definition for call implementation."""
        return self._call_impl(features)


def get_params_per_level(
    cls_out: torch.Tensor,
    reg_out: torch.Tensor,
    anchors: torch.Tensor,
    num_pre_nms: int = 2000,
    score_thr: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get topk params from feature output per level per image before nms.

    Params include flattened classification scores, box energies, and
    corresponding anchors.

    Args:
        cls_out (torch.Tensor):
            [C, H, W] classification scores at a particular scale.
        reg_out (torch.Tensor):
            [C, H, W] regression parameters at a particular scale.
        anchors (torch.Tensor): [H * W, 4] anchor boxes per cell.
        num_pre_nms (int): number of predictions before nms.
        score_thr (float): score threshold for filtering predictions.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: topk
            flattened classification, regression outputs, and corresponding
            anchors.
    """
    assert cls_out.size()[-2:] == reg_out.size()[-2:], (
        f"Shape mismatch: cls_out({cls_out.size()[-2:]}), reg_out("
        f"{reg_out.size()[-2:]})."
    )
    reg_out = reg_out.permute(1, 2, 0).reshape(-1, 4)
    cls_out = cls_out.permute(1, 2, 0).reshape(reg_out.size(0), -1).sigmoid()
    valid_mask = torch.greater(cls_out, score_thr)
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(num_pre_nms, valid_idxs.size(0))
    cls_out_filt = cls_out[valid_mask]
    cls_out_ranked, rank_inds = cls_out_filt.sort(descending=True)
    topk_inds = valid_idxs[rank_inds[:num_topk]]
    keep_inds, labels = topk_inds.unbind(dim=1)
    cls_out = cls_out_ranked[:num_topk]
    reg_out = reg_out[keep_inds, :]
    anchors = anchors[keep_inds, :]

    return cls_out, labels, reg_out, anchors


def decode_multi_level_outputs(
    cls_out_all: list[torch.Tensor],
    lbl_out_all: list[torch.Tensor],
    reg_out_all: list[torch.Tensor],
    anchors_all: list[torch.Tensor],
    image_hw: tuple[int, int],
    box_decoder: DeltaXYWHBBoxDecoder,
    max_per_img: int = 1000,
    nms_threshold: float = 0.7,
    min_box_size: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode box energies into detections for a single image.

    Detections are post-processed via NMS. NMS is performed per level.
    Afterwards, select topk detections.

    Args:
        cls_out_all (list[torch.Tensor]): topk class scores per level.
        lbl_out_all (list[torch.Tensor]): topk class labels per level.
        reg_out_all (list[torch.Tensor]): topk regression params per level.
        anchors_all (list[torch.Tensor]): topk anchor boxes per level.
        image_hw (tuple[int, int]): image size.
        box_decoder (DeltaXYWHBBoxDecoder): bounding box encoder.
        max_per_img (int, optional): maximum predictions per image.
            Defaults to 1000.
        nms_threshold (float, optional): iou threshold for NMS.
            Defaults to 0.7.
        min_box_size (tuple[int, int], optional): minimum box size.
            Defaults to (0, 0).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: decoded proposal boxes & scores.
    """
    scores, labels = torch.cat(cls_out_all), torch.cat(lbl_out_all)
    boxes = bbox_clip(
        box_decoder(torch.cat(anchors_all), torch.cat(reg_out_all)),
        image_hw,
    )

    boxes, mask = filter_boxes_by_area(boxes, min_area=prod(min_box_size))
    scores, labels = scores[mask], labels[mask]

    if boxes.numel() > 0:
        keep = batched_nms(boxes, scores, labels, iou_threshold=nms_threshold)[
            :max_per_img
        ]
        return boxes[keep], scores[keep], labels[keep]
    return (boxes.new_zeros(0, 4), scores.new_zeros(0), labels.new_zeros(0))


class Dense2Det(nn.Module):
    """Compute detections from dense network outputs.

    This class acts as a stateless functor that does the following:
    1. Create anchor grid for feature grids (classification and regression
    outputs) at all scales.
    For each image
        For each level
            2. Get a topk pre-selection of flattened classification scores and
            box energies from feature output before NMS.
        3. Decode class scores and box energies into detection boxes,
        apply NMS.
    Return detection boxes for all images.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_decoder: DeltaXYWHBBoxDecoder,
        num_pre_nms: int = 2000,
        max_per_img: int = 1000,
        nms_threshold: float = 0.7,
        min_box_size: tuple[int, int] = (0, 0),
        score_thr: float = 0.0,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_decoder = box_decoder
        self.num_pre_nms = num_pre_nms
        self.max_per_img = max_per_img
        self.nms_threshold = nms_threshold
        self.min_box_size = min_box_size
        self.score_thr = score_thr

    def forward(
        self,
        cls_outs: list[torch.Tensor],
        reg_outs: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Compute detections from dense network outputs.

        Generate anchor grid for all scales.
        For each batch element:
            Compute classification, regression, and anchor pairs for all
            scales. Decode those pairs into proposals, post-process with NMS.

        Args:
            cls_outs (list[torch.Tensor]): [N, C * A, H, W] per scale.
            reg_outs (list[torch.Tensor]): [N, 4 * A, H, W] per scale.
            images_hw (list[tuple[int, int]]): list of image sizes.

        Returns:
            DetOut: Detection outputs.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = cls_outs[0].device
        featmap_sizes: list[tuple[int, int]] = [
            featmap.size()[-2:] for featmap in cls_outs  # type: ignore
        ]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals, scores, labels = [], [], []
        for img_id, image_hw in enumerate(images_hw):
            cls_out_all, lbl_out_all, reg_out_all, anchors_all = [], [], [], []
            for cls_out, reg_out, anchor_grid in zip(
                cls_outs, reg_outs, anchor_grids
            ):
                cls_out_, lbl_out, reg_out_, anchors = get_params_per_level(
                    cls_out[img_id],
                    reg_out[img_id],
                    anchor_grid,
                    self.num_pre_nms,
                    self.score_thr,
                )
                cls_out_all += [cls_out_]
                lbl_out_all += [lbl_out]
                reg_out_all += [reg_out_]
                anchors_all += [anchors]

            box, score, label = decode_multi_level_outputs(
                cls_out_all,
                lbl_out_all,
                reg_out_all,
                anchors_all,
                image_hw,
                self.box_decoder,
                self.max_per_img,
                self.nms_threshold,
                self.min_box_size,
            )
            proposals.append(box)
            scores.append(score)
            labels.append(label)
        return DetOut(proposals, scores, labels)

    def __call__(
        self,
        cls_outs: list[torch.Tensor],
        reg_outs: list[torch.Tensor],
        images_hw: list[tuple[int, int]],
    ) -> DetOut:
        """Type definition for function call."""
        return self._call_impl(cls_outs, reg_outs, images_hw)


class RetinaNetHeadLoss(DenseAnchorHeadLoss):
    """Loss of RetinaNet head."""

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        box_matcher: None | Matcher = None,
        box_sampler: None | Sampler = None,
        loss_cls: TorchLossFunc = sigmoid_focal_loss,
        loss_bbox: TorchLossFunc = l1_loss,
    ) -> None:
        """Creates an instance of the class.

        Args:
            anchor_generator (AnchorGenerator): Generates anchor grid priors.
            box_encoder (DeltaXYWHBBoxEncoder): Encodes bounding boxes to the
                desired network output.
            box_matcher (None | Matcher, optional): Box matcher. Defaults to
                None.
            box_sampler (None | Sampler, optional): Box sampler. Defaults to
                None.
            loss_cls (TorchLossFunc, optional): Classification loss function.
                Defaults to sigmoid_focal_loss.
            loss_bbox (TorchLossFunc, optional): Regression loss function.
                Defaults to l1_loss.
        """
        matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
        )
        sampler = (
            box_sampler
            if box_sampler is not None
            else get_default_box_sampler()
        )
        super().__init__(
            anchor_generator,
            box_encoder,
            matcher,
            sampler,
            loss_cls,
            loss_bbox,
        )
