"""RetinaNet."""
from math import prod
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import nn
from torchvision.ops import batched_nms

from vis4d.op.box.encoder import BaseBoxEncoder2D, DeltaXYWHBBoxEncoder
from vis4d.op.box.matchers import BaseMatcher, MaxIoUMatcher
from vis4d.struct_to_revise.labels.boxes import filter_boxes

from .anchor_generator import AnchorGenerator
from .rcnn import DetOut


class RetinaNetOut(NamedTuple):
    """RetinaNet head outputs."""

    # logits for the box classication for each feature level. The logit
    # dimention is number of classes plus 1 for the background.
    cls_score: List[torch.Tensor]
    # Each box has regression for all classes for each feature level. So the
    # tensor dimention is [batch_size, number of boxes, number of classes x 4]
    bbox_pred: List[torch.Tensor]


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1.0, 2.0],
        strides=[8, 16, 32, 64, 128],
    )


def get_default_box_encoder() -> DeltaXYWHBBoxEncoder:
    """Get the default bounding box encoder."""
    return DeltaXYWHBBoxEncoder(
        target_means=(0.0, 0.0, 0.0, 0.0), target_stds=(1.0, 1.0, 1.0, 1.0)
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default bounding box matcher."""
    return MaxIoUMatcher(
        thresholds=[0.4, 0.5],
        labels=[0, -1, 1],
        allow_low_quality_matches=True,
    )


class RetinaNetHead(nn.Module):
    """RetinaNet Head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_sigmoid_cls: bool = True,
        anchor_generator: Optional[AnchorGenerator] = None,
        box_encoder: Optional[BaseBoxEncoder2D] = None,
        box_matcher: Optional[BaseMatcher] = None,
    ):
        """Init."""
        super().__init__()
        self.anchor_generator = (
            anchor_generator
            if anchor_generator is not None
            else get_default_anchor_generator()
        )
        self.box_encoder = (
            box_encoder
            if box_encoder is not None
            else get_default_box_encoder()
        )
        self.box_matcher = (
            box_matcher
            if box_matcher is not None
            else get_default_box_matcher()
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

    def forward(self, features: List[torch.Tensor]) -> RetinaNetOut:
        """RetinaNet forward.

        Args:
            features (List[torch.Tensor]): Feature pyramid

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

    def __call__(self, features: List[torch.Tensor]) -> RetinaNetOut:
        """Type definition for call implementation."""
        return self._call_impl(features)


class Dense2Det(nn.Module):
    """Compute detections from dense network outputs.

    This class acts as a stateless functor that does the following:
    1. Create anchor grid for feature grids (classification and regression outputs) at all scales.
    For each image
        For each level
            2. Get a topk pre-selection of flattened classification scores and box energies from feature output before NMS.
        3. Decode class scores and box energies into detection boxes, apply NMS.
    Return detection boxes for all images.
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        box_encoder: DeltaXYWHBBoxEncoder,
        num_pre_nms: int = 2000,
        max_per_img: int = 1000,
        nms_threshold: float = 0.7,
        min_box_size: Tuple[int, int] = (0, 0),
        score_thr: float = 0.0,
    ) -> None:
        """Init."""
        super().__init__()
        self.anchor_generator = anchor_generator
        self.box_encoder = box_encoder
        self.num_pre_nms = num_pre_nms
        self.max_per_img = max_per_img
        self.nms_threshold = nms_threshold
        self.min_box_size = min_box_size
        self.score_thr = score_thr

    def _get_params_per_level(
        self,
        cls_out: torch.Tensor,
        reg_out: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a topk pre-selection of flattened classification scores and box
        energies from feature output per level per image before nms.

        Args:
            cls_out (torch.Tensor): [C, H, W] classification scores at a particular scale.
            reg_out (torch.Tensor): [C, H, W] regression parameters at a particular scale.
            anchors (torch.Tensor): [H*W, 4] anchor boxes per cell.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: topk
                flattened classification, regression outputs, and corresponding anchors.
        """
        assert cls_out.size()[-2:] == reg_out.size()[-2:], (
            f"Shape mismatch: cls_out({cls_out.size()[-2:]}), reg_out("
            f"{reg_out.size()[-2:]})."
        )
        reg_out = reg_out.permute(1, 2, 0).reshape(-1, 4)
        cls_out = (
            cls_out.permute(1, 2, 0).reshape(reg_out.size(0), -1).sigmoid()
        )
        valid_mask = cls_out > self.score_thr
        valid_idxs = torch.nonzero(valid_mask)
        num_topk = min(self.num_pre_nms, valid_idxs.size(0))
        cls_out_filt = cls_out[valid_mask]
        cls_out_ranked, rank_inds = cls_out_filt.sort(descending=True)
        topk_inds = valid_idxs[rank_inds[:num_topk]]
        keep_inds, labels = topk_inds.unbind(dim=1)
        cls_out = cls_out_ranked[:num_topk]
        reg_out = reg_out[keep_inds, :]
        anchors = anchors[keep_inds, :]

        return cls_out, labels, reg_out, anchors

    def _decode_multi_level_outputs(
        self,
        cls_out_all: List[torch.Tensor],
        lbl_out_all: List[torch.Tensor],
        reg_out_all: List[torch.Tensor],
        anchors_all: List[torch.Tensor],
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode box energies into detections for a single image, post-process
        via NMS. NMS is performed per level. Afterwards, select topk detections.

        Args:
            cls_out_all (List[torch.Tensor]): topk class scores per level.
            lbl_out_all (List[torch.Tensor]): topk class labels per level.
            reg_out_all (List[torch.Tensor]): topk regression params per level.
            anchors_all (List[torch.Tensor]): topk anchor boxes per level.
            image_hw (Tuple[int, int]): image size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: decoded proposal boxes & scores.
        """
        scores, labels = torch.cat(cls_out_all), torch.cat(lbl_out_all)
        boxes = self.box_encoder.decode(
            torch.cat(anchors_all), torch.cat(reg_out_all), max_shape=image_hw
        )

        boxes, mask = filter_boxes(boxes, min_area=prod(self.min_box_size))
        scores, labels = scores[mask], labels[mask]

        if boxes.numel() > 0:
            keep = batched_nms(
                boxes, scores, labels, iou_threshold=self.nms_threshold
            )[: self.max_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        else:
            return (
                boxes.new_zeros(0, 4),
                scores.new_zeros(0),
                labels.new_zeros(0),
            )
        return boxes, scores, labels

    def forward(
        self,
        class_outs: List[torch.Tensor],
        regression_outs: List[torch.Tensor],
        images_hw: List[Tuple[int, int]],
    ) -> DetOut:
        """Compute detections from dense network outputs.

        Generate anchor grid for all scales.
        For each batch element:
            Compute classification, regression and anchor pairs for all scales.
            Decode those pairs into proposals, post-process with NMS.

        Args:
            class_outs (List[torch.Tensor]): [N, 1 * A, H, W] per scale.
            regression_outs (List[torch.Tensor]): [N, 4 * A, H, W] per scale.
            images_hw (List[Tuple[int, int]]): list of image sizes.

        Returns:
            DetOut: detection outputs.
        """
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        device = class_outs[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in class_outs]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        anchor_grids = self.anchor_generator.grid_priors(
            featmap_sizes, device=device
        )
        proposals, scores, labels = [], [], []
        for img_id, image_hw in enumerate(images_hw):
            cls_out_all, lbl_out_all, reg_out_all, anchors_all = [], [], [], []
            for cls_outs, reg_outs, anchor_grid in zip(
                class_outs, regression_outs, anchor_grids
            ):
                (
                    cls_out,
                    lbl_out,
                    reg_out,
                    anchors,
                ) = self._get_params_per_level(
                    cls_outs[img_id], reg_outs[img_id], anchor_grid
                )
                cls_out_all += [cls_out]
                lbl_out_all += [lbl_out]
                reg_out_all += [reg_out]
                anchors_all += [anchors]

            box, score, label = self._decode_multi_level_outputs(
                cls_out_all, lbl_out_all, reg_out_all, anchors_all, image_hw
            )
            proposals.append(box)
            scores.append(score)
            labels.append(label)
        return DetOut(proposals, scores, labels)
