"""mmdetection detector wrapper."""
from typing import Dict, List, Optional, Sequence, Tuple, Union

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.model.mmdet_utils import get_mmdet_config
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    ModelOutput,
)

from ..backbone import MMDetBackboneConfig, build_backbone
from ..backbone.neck import MMDetNeckConfig
from ..base import BaseModelConfig
from ..heads.dense_head import MMDetDenseHeadConfig, build_dense_head
from ..heads.roi_head import MMDetRoIHeadConfig, build_roi_head
from .base import BaseDetectorConfig, BaseTwoStageDetector
from .utils import postprocess

try:
    from mmcv.runner.checkpoint import load_checkpoint

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False

MMDET_MODEL_PREFIX = "https://download.openmmlab.com/mmdetection/v2.0/"
REV_KEYS = [
    ("roi_head", "roi_head.mm_roi_head"),
    ("rpn_head", "rpn_head.mm_dense_head"),
    ("backbone", "backbone.mm_backbone"),
    ("neck", "backbone.neck.mm_neck"),
]


class MMTwoStageDetectorConfig(BaseDetectorConfig):
    """Config for mmdetection two stage models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    backbone_output_names: Optional[List[str]]
    weights: Optional[str]


class MMTwoStageDetector(BaseTwoStageDetector):
    """mmdetection two-stage detector wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMTwoStageDetector requires both mmcv and mmdet to be installed!"
        super().__init__(cfg)
        self.cfg: MMTwoStageDetectorConfig = MMTwoStageDetectorConfig(
            **cfg.dict()
        )
        self.mm_cfg = get_mmdet_config(self.cfg)
        self.backbone = build_backbone(
            MMDetBackboneConfig(
                type="MMDetBackbone",
                mm_cfg=self.mm_cfg["backbone"],
                pixel_mean=self.cfg.pixel_mean,
                pixel_std=self.cfg.pixel_std,
                neck=MMDetNeckConfig(
                    type="MMDetNeck",
                    mm_cfg=self.mm_cfg["neck"],
                    output_names=self.cfg.backbone_output_names,
                ),
            )
        )

        rpn_cfg = self.mm_cfg["rpn_head"]
        if "train_cfg" in self.mm_cfg and "rpn" in self.mm_cfg["train_cfg"]:
            rpn_train_cfg = self.mm_cfg["train_cfg"]["rpn"]
        else:
            rpn_train_cfg = None
        rpn_cfg.update(
            train_cfg=rpn_train_cfg, test_cfg=self.mm_cfg["test_cfg"]["rpn"]
        )
        self.rpn_head = build_dense_head(
            MMDetDenseHeadConfig(type="MMDetDenseHead", mm_cfg=rpn_cfg)
        )

        roi_head_cfg = self.mm_cfg["roi_head"]
        if "train_cfg" in self.mm_cfg and "rcnn" in self.mm_cfg["train_cfg"]:
            rcnn_train_cfg = self.mm_cfg["train_cfg"]["rcnn"]
        else:
            rcnn_train_cfg = None

        roi_head_cfg.update(train_cfg=rcnn_train_cfg)
        roi_head_cfg.update(test_cfg=self.mm_cfg["test_cfg"]["rcnn"])
        self.roi_head = build_roi_head(
            MMDetRoIHeadConfig(
                type="MMDetRoIHead",
                mm_cfg=roi_head_cfg,
            )
        )

        self.with_mask = self.roi_head.with_mask
        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmdet://"):
                self.cfg.weights = (
                    MMDET_MODEL_PREFIX + self.cfg.weights.split("mmdet://")[-1]
                )
            load_checkpoint(self, self.cfg.weights, revise_keys=REV_KEYS)

        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMTwoStageDetector training!"
        inputs = batch_inputs[0]
        features = self.backbone(inputs)
        proposals, rpn_losses = self.rpn_head(inputs, features, inputs.targets)
        roi_losses, _ = self.roi_head(
            inputs, proposals, features, inputs.targets
        )
        return {**rpn_losses, **roi_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMTwoStageDetector testing!"
        inputs = batch_inputs[0]

        features = self.backbone(inputs)
        proposals = self.rpn_head(inputs, features)
        outs = self.roi_head(inputs, proposals, features)
        detections, segmentations = [o[0] for o in outs], [o[1] for o in outs]

        postprocess(
            inputs, detections, segmentations, self.cfg.clip_bboxes_to_image
        )
        outputs = dict(
            detect=[d.to_scalabel(self.cat_mapping) for d in detections]
        )
        if self.with_mask:
            assert segmentations is not None
            segmentations: List[InstanceMasks]  # type: ignore
            outputs.update(
                ins_seg=[
                    s.to_scalabel(self.cat_mapping) for s in segmentations
                ]
            )
        return outputs

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Detector feature extraction stage.

        Return backbone output features.
        """
        return self.backbone(inputs)

    def generate_proposals(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[Tuple[LossesType, List[Boxes2D]], List[Boxes2D]]:
        """Detector RPN stage.

        Return proposals per image and losses (empty if no targets).
        """
        return self.rpn_head(inputs, features, targets)

    def generate_detections(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
        targets: Optional[LabelInstances] = None,
    ) -> Union[
        Tuple[LossesType, Optional[SamplingResult]],
        Sequence[Tuple[Boxes2D, Optional[InstanceMasks]]],
    ]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        return self.roi_head(inputs, proposals, features, targets)
