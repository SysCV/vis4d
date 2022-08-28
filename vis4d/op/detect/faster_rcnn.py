"""Faster RCNN detector."""
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from vis4d.common.bbox.anchor_generator import AnchorGenerator
from vis4d.common.bbox.coders.delta_xywh_coder import DeltaXYWHBBoxEncoder
from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import (
    RandomSampler,
    match_and_sample_proposals,
)
from vis4d.op.heads.dense_head.rpn import TransformRPNOutputs
from vis4d.struct import Boxes2D, LossesType

from ..heads.dense_head import RPNHead
from ..heads.roi_head.rcnn import RCNNHead


class FRCNNReturn(NamedTuple):
    rpn_cls_out: torch.Tensor
    rpn_reg_out: torch.Tensor
    # rpn: Optional[NamedTuple]  # TODO define
    roi_cls_out: torch.Tensor
    roi_reg_out: torch.Tensor
    proposal_boxes: List[torch.Tensor]
    proposal_scores: List[torch.Tensor]
    proposal_target_boxes: Optional[List[torch.Tensor]]
    proposal_target_classes: Optional[List[torch.Tensor]]
    proposal_labels: Optional[List[torch.Tensor]]


def get_default_anchor_generator() -> AnchorGenerator:
    """Get default anchor generator."""
    return AnchorGenerator(
        scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
    )


def get_default_rpn_box_encoder() -> DeltaXYWHBBoxEncoder:
    """Get the default bounding box encoder for RPN."""
    return DeltaXYWHBBoxEncoder(
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
    )


def get_default_rcnn_box_encoder() -> DeltaXYWHBBoxEncoder:
    """Get the default bounding box encoder for RCNN."""
    return DeltaXYWHBBoxEncoder(
        clip_border=True,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
    )


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default bounding box matcher."""
    return MaxIoUMatcher(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    )


def get_default_box_sampler() -> RandomSampler:
    """Get default bounding box sampler."""
    return RandomSampler(batch_size=512, positive_fraction=0.25)


class FasterRCNN(nn.Module):
    """mmdetection two-stage detector wrapper."""

    def __init__(
        self,
        num_classes: int = 80,
        anchor_generator: Optional[AnchorGenerator] = None,
        rpn_box_encoder: Optional[DeltaXYWHBBoxEncoder] = None,
        rcnn_box_encoder: Optional[DeltaXYWHBBoxEncoder] = None,
        box_matcher: Optional[MaxIoUMatcher] = None,
        box_sampler: Optional[RandomSampler] = None,
    ):
        """Init."""
        super().__init__()
        self.anchor_generator = (
            anchor_generator
            if anchor_generator is not None
            else get_default_anchor_generator()
        )
        self.rpn_box_encoder = (
            rpn_box_encoder
            if rpn_box_encoder is not None
            else get_default_rpn_box_encoder()
        )
        self.rcnn_box_encoder = (
            rcnn_box_encoder
            if rcnn_box_encoder is not None
            else get_default_rcnn_box_encoder()
        )
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

        self.rpn_head = RPNHead(self.anchor_generator.num_base_priors[0])
        self.rpn_head_transform = TransformRPNOutputs(
            self.anchor_generator, self.rpn_box_encoder
        )
        self.roi_head = RCNNHead(num_classes=num_classes)

    def forward(
        self,
        features: List[torch.Tensor],
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> FRCNNReturn:
        """Forward pass during training stage.

        TODO(tobiasfshr) consider indiviual image sizes and paddings to
        remove invalid proposals.

        Returns:
            rpn_class_out
            rpn_box_out
            proposals

        """
        if target_boxes is not None:
            assert target_classes is not None

        rpn_cls_out, rpn_reg_out = self.rpn_head(features[2:])
        proposals, scores = self.rpn_head_transform(
            rpn_cls_out, rpn_reg_out, features[0].shape
        )

        if target_boxes is not None:
            (
                proposals,
                scores,
                sampled_target_boxes,
                sampled_target_classes,
                sampled_labels,
            ) = match_and_sample_proposals(
                self.box_matcher,
                self.box_sampler,
                proposals,
                scores,
                target_boxes,
                target_classes,
                proposal_append_gt=True,
            )

        else:
            sampled_target_boxes, sampled_target_classes, sampled_labels = (
                None,
                None,
                None,
            )

        roi_cls_out, roi_reg_out = self.roi_head(features[2:], proposals)

        return FRCNNReturn(
            rpn_cls_out=rpn_cls_out,
            rpn_reg_out=rpn_reg_out,
            roi_reg_out=roi_reg_out,
            roi_cls_out=roi_cls_out,
            proposal_boxes=proposals,
            proposal_scores=scores,
            proposal_target_boxes=sampled_target_boxes,
            proposal_target_classes=sampled_target_classes,
            proposal_labels=sampled_labels,
        )

    def __call__(
        self,
        features: torch.Tensor,
        target_boxes: Optional[List[torch.Tensor]] = None,
        target_classes: Optional[List[torch.Tensor]] = None,
    ) -> FRCNNReturn:
        """Type definition for call implementation."""
        return self._call_impl(features, target_boxes, target_classes)


class FasterRCNNLoss(nn.Module):  # TODO needs to be updated / removed
    def __init__(self, rpn_head, roi_head):
        super().__init__()
        from vis4d.op.heads.dense_head.rpn import MMDetDenseHeadLoss
        from vis4d.op.heads.roi_head.rcnn import RCNNLoss

        self.rpn_head_loss = MMDetDenseHeadLoss(rpn_head)
        self.roi_head_loss = RCNNLoss(roi_head)

    def forward(self, frcnn_returns: FRCNNReturn, targets) -> LossesType:
        losses = self.rpn_head_loss(
            frcnn_returns.rpn_cls_out,
            frcnn_returns.rpn_reg_out,
            target_classes,
            images_shape,
        )


# class MMOneStageDetector(BaseOneStageDetector):
#     """mmdetection one-stage detector wrapper."""
#
#     def __init__(
#         self,
#         *args: ArgsType,
#         pixel_mean: Optional[Tuple[float, float, float]] = None,
#         pixel_std: Optional[Tuple[float, float, float]] = None,
#         model_base: Optional[str] = None,
#         model_kwargs: Optional[DictStrAny] = None,
#         backbone_output_names: Optional[List[str]] = None,
#         weights: Optional[str] = None,
#         backbone: Optional[BaseBackbone] = None,
#         bbox_head = None,
#         **kwargs: ArgsType,
#     ):
#         """Init."""
#         assert (
#             MMDET_INSTALLED and MMCV_INSTALLED
#         ), "MMTwoStageDetector requires both mmcv and mmdet to be installed!"
#         super().__init__(*args, **kwargs)
#         assert self.category_mapping is not None
#         self.cat_mapping = {v: k for k, v in self.category_mapping.items()}
#         if backbone is None or bbox_head is None:
#             assert model_base is not None
#             self.mm_cfg = get_mmdet_config(
#                 model_base, model_kwargs, self.category_mapping
#             )
#         if pixel_mean is None or pixel_std is None:
#             assert backbone is not None, (
#                 "If no custom backbone is defined, image "
#                 "normalization parameters must be specified!"
#             )
#
#         if backbone is None:
#             self.backbone: BaseBackbone = MMDetBackbone(
#                 mm_cfg=self.mm_cfg["backbone"],
#                 pixel_mean=pixel_mean,
#                 pixel_std=pixel_std,
#                 neck=MMDetNeck(
#                     mm_cfg=self.mm_cfg["neck"],
#                     output_names=backbone_output_names,
#                 ),
#             )
#         else:
#             self.backbone = backbone
#
#         if bbox_head is None:
#             bbox_cfg = self.mm_cfg["bbox_head"]
#             if "train_cfg" in self.mm_cfg:
#                 bbox_train_cfg = self.mm_cfg["train_cfg"]
#             else:  # pragma: no cover
#                 bbox_train_cfg = None
#             bbox_cfg.update(
#                 train_cfg=bbox_train_cfg,
#                 test_cfg=self.mm_cfg["test_cfg"],
#             )
#             self.bbox_head = MMDetDenseHead(
#                 mm_cfg=bbox_cfg, category_mapping=self.category_mapping
#             )
#         else:
#             self.bbox_head = bbox_head
#
#         if weights is not None:
#             load_model_checkpoint(self, weights, REV_KEYS)
#
#     def forward(
#         self, batch_inputs: List[InputSample]
#     ) -> Union[LossesType, ModelOutput]:
#         """Forward."""
#         if self.training:
#             return self.forward_train(batch_inputs)
#         return self.forward_test(batch_inputs)
#
#     def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
#         """Forward pass during training stage."""
#         assert (
#             len(batch_inputs) == 1
#         ), "No reference views allowed in MMOneStageDetector training!"
#         inputs, targets = batch_inputs[0], batch_inputs[0].targets
#         assert targets is not None, "Training requires targets."
#         features = self.backbone(inputs)
#         bbox_losses, _ = self.bbox_head(inputs, features, targets)
#         return {**bbox_losses}
#
#     def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
#         """Forward pass during testing stage."""
#         assert (
#             len(batch_inputs) == 1
#         ), "No reference views allowed in MMOneStageDetector testing!"
#         inputs = batch_inputs[0]
#         features = self.backbone(inputs)
#         detections = self.bbox_head(inputs, features)
#         outputs = dict(detect=detections)
#         postprocess_predictions(inputs, outputs, self.clip_bboxes_to_image)
#         return predictions_to_scalabel(outputs, self.cat_mapping)
#
#     def extract_features(self, inputs: InputSample) -> NamedTensors:
#         """Detector feature extraction stage.
#
#         Return backbone output features.
#         """
#         feats = self.backbone(inputs)
#         assert isinstance(feats, dict)
#         return feats
#
#     def _detections_train(
#         self,
#         inputs: InputSample,
#         features: NamedTensors,
#         targets: LabelInstances,
#     ) -> Tuple[LossesType, Optional[List[Boxes2D]]]:
#         """Train stage detections generation."""
#         return self.bbox_head(inputs, features, targets)
#
#     def _detections_test(
#         self, inputs: InputSample, features: NamedTensors
#     ) -> List[Boxes2D]:
#         """Test stage detections generation."""
#         return self.bbox_head(inputs, features)

#
# def get_mmdet_config(
#     model_base: str,
#     model_kwargs: Optional[DictStrAny] = None,
#     category_mapping: Optional[Dict[str, int]] = None,
# ) -> MMConfig:
#     """Convert a Detector config to a mmdet readable config."""
#     cfg = load_config(model_base)
#
#     # convert detect attributes
#     if category_mapping is not None:
#         if "bbox_head" in cfg:  # pragma: no cover
#             cfg["bbox_head"]["num_classes"] = len(category_mapping)
#         if "roi_head" in cfg:
#             cfg["roi_head"]["bbox_head"]["num_classes"] = len(category_mapping)
#             if "mask_head" in cfg["roi_head"]:
#                 cfg["roi_head"]["mask_head"]["num_classes"] = len(
#                     category_mapping
#                 )
#
#     if model_kwargs is not None:
#         add_keyword_args(model_kwargs, cfg)
#     return cfg
