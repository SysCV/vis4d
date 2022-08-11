"""Faster RCNN detector."""
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common.bbox.matchers import MaxIoUMatcher
from vis4d.common.bbox.samplers import (
    RandomSampler,
    SamplingResult,
    match_and_sample_proposals,
)
from vis4d.model.heads.dense_head.rpn import TransformRPNOutputs
from vis4d.struct import (
    ArgsType,
    Boxes2D,
    DictStrAny,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    NamedTensors,
)
from vis4d.struct.labels.boxes import tensor_to_boxes2d

from ..backbone import BaseBackbone, MMDetBackbone
from ..backbone.neck import MMDetNeck
from ..heads.dense_head import RPNHead
from ..heads.roi_head.rcnn import Box2DRoIHead
from ..utils import (
    add_keyword_args,
    load_config,
    load_model_checkpoint,
    postprocess_predictions,
    predictions_to_scalabel,
)

REV_KEYS = [
    (r"^rpn_head.rpn_reg\.", "rpn_head.rpn_box."),
    (r"^roi_head.bbox_head\.", "roi_head."),
    (r"^backbone\.", "backbone.body."),
    (r"^neck.lateral_convs\.", "backbone.fpn.inner_blocks."),
    (r"^neck.fpn_convs\.", "backbone.fpn.layer_blocks."),
    ("\.conv.weight", ".weight"),
    ("\.conv.bias", ".bias"),
]


class FRCNNReturn(NamedTuple):
    rpn_cls_out: torch.Tensor
    rpn_reg_out: torch.Tensor
    roi_cls_out: torch.Tensor
    roi_reg_out: torch.Tensor
    proposal_boxes: List[torch.Tensor]
    proposal_scores: List[torch.Tensor]


class FasterRCNN(nn.Module):
    """mmdetection two-stage detector wrapper."""

    def __init__(
        self,
        backbone: nn.Module,
        weights: Optional[str] = None,
    ):
        """Init."""
        super().__init__()
        self.backbone = backbone

        self.rpn_head = RPNHead()
        self.rpn_head_transform = TransformRPNOutputs()

        self.bbox_matcher = MaxIoUMatcher(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        )
        self.bbox_sampler = RandomSampler(
            batch_size_per_image=512, positive_fraction=0.25
        )

        self.roi_head = Box2DRoIHead()

        if weights is not None:
            load_model_checkpoint(self, weights, REV_KEYS)

    def forward(
        self,
        images: torch.Tensor,
        target_boxes: Optional[torch.Tensor] = None,
        target_classes=None,
    ) -> FRCNNReturn:
        """Forward pass during training stage.

        Returns:
            rpn_class_out
            rpn_box_out
            proposals

        """

        if target_boxes is not None:
            assert target_classes is not None

        features = self.backbone(images)

        rpn_cls_out, rpn_reg_out = self.rpn_head(features)
        proposals = self.rpn_head_transform(
            rpn_cls_out, rpn_reg_out, images.shape
        )

        if target_boxes is not None:
            sampling_result = match_and_sample_proposals(
                self.bbox_matcher,
                self.bbox_sampler,
                proposals,
                tensor_to_boxes2d(target_boxes),
                proposal_append_gt=True,
            )
            proposals = sampling_result.sampled_boxes
        roi_cls_out, roi_reg_out = self.roi_head(
            features[:-1], [p.boxes for p in proposals]
        )

        return FRCNNReturn(
            rpn_cls_out=rpn_cls_out,
            rpn_reg_out=rpn_reg_out,
            roi_reg_out=roi_reg_out,
            roi_cls_out=roi_cls_out,
            proposal_boxes=[p.boxes for p in proposals],
            proposal_scores=[p.score for p in proposals],
        )


class FasterRCNNLoss(nn.Module):
    def __init__(self, rpn_head, roi_head):
        super().__init__()
        from vis4d.model.heads.dense_head.rpn import MMDetDenseHeadLoss
        from vis4d.model.heads.roi_head.rcnn import MMDetFRCNNRoIHeadLoss

        self.rpn_head_loss = MMDetDenseHeadLoss(rpn_head)
        self.roi_head_loss = MMDetFRCNNRoIHeadLoss(roi_head)

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
