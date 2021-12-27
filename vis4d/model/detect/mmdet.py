"""mmdetection detector wrapper."""
from typing import Dict, List, Optional, Tuple, Union

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    ModelOutput,
    TLabelInstance,
)

from ..backbone import BaseBackboneConfig, MMDetBackboneConfig, build_backbone
from ..backbone.neck import MMDetNeckConfig
from ..base import BaseModelConfig
from ..heads.dense_head import (
    BaseDenseHead,
    BaseDenseHeadConfig,
    MMDetDenseHeadConfig,
    build_dense_head,
)
from ..heads.roi_head import (
    BaseRoIHead,
    BaseRoIHeadConfig,
    MMDetRoIHeadConfig,
    build_roi_head,
)
from ..mmdet_utils import add_keyword_args, load_config
from ..utils import predictions_to_scalabel
from .base import (
    BaseDetectorConfig,
    BaseOneStageDetector,
    BaseTwoStageDetector,
)

try:
    from mmcv import Config as MMConfig
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
    (r"^roi_head\.", "roi_head.mm_roi_head."),
    (r"^rpn_head\.", "rpn_head.mm_dense_head."),
    (r"^bbox_head\.", "bbox_head.mm_dense_head."),
    (r"^backbone\.", "backbone.mm_backbone."),
    (r"^neck\.", "backbone.neck.mm_neck."),
]


class MMTwoStageDetectorConfig(BaseDetectorConfig):
    """Config for mmdetection two stage models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    backbone_output_names: Optional[List[str]]
    weights: Optional[str]
    backbone: Optional[BaseBackboneConfig]
    roi_head: Optional[BaseRoIHeadConfig]
    rpn_head: Optional[BaseDenseHeadConfig]


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
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.mm_cfg = get_mmdet_config(self.cfg)
        if self.cfg.backbone is None:
            self.cfg.backbone = MMDetBackboneConfig(
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
        self.backbone = build_backbone(self.cfg.backbone)

        if self.cfg.rpn_head is None:
            rpn_cfg = self.mm_cfg["rpn_head"]
            if (
                "train_cfg" in self.mm_cfg
                and "rpn" in self.mm_cfg["train_cfg"]
            ):
                rpn_train_cfg = self.mm_cfg["train_cfg"]["rpn"]
            else:  # pragma: no cover
                rpn_train_cfg = None
            rpn_cfg.update(
                train_cfg=rpn_train_cfg,
                test_cfg=self.mm_cfg["test_cfg"]["rpn"],
            )
            self.cfg.rpn_head = MMDetDenseHeadConfig(
                type="MMDetRPNHead",
                mm_cfg=rpn_cfg,
                category_mapping=self.cfg.category_mapping,
            )
        self.rpn_head: BaseDenseHead[
            List[Boxes2D], List[Boxes2D]
        ] = build_dense_head(self.cfg.rpn_head)

        if self.cfg.roi_head is None:
            roi_head_cfg = self.mm_cfg["roi_head"]
            if (
                "train_cfg" in self.mm_cfg
                and "rcnn" in self.mm_cfg["train_cfg"]
            ):
                rcnn_train_cfg = self.mm_cfg["train_cfg"]["rcnn"]
            else:  # pragma: no cover
                rcnn_train_cfg = None

            roi_head_cfg.update(train_cfg=rcnn_train_cfg)
            roi_head_cfg.update(test_cfg=self.mm_cfg["test_cfg"]["rcnn"])
            self.cfg.roi_head = MMDetRoIHeadConfig(
                type="MMDetRoIHead",
                mm_cfg=roi_head_cfg,
                category_mapping=self.cfg.category_mapping,
            )
        self.roi_head: BaseRoIHead[
            Optional[SamplingResult],
            Tuple[List[Boxes2D], Optional[List[InstanceMasks]]],
        ] = build_roi_head(self.cfg.roi_head)

        self.with_mask = self.roi_head.with_mask
        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmdet://"):
                self.cfg.weights = (
                    MMDET_MODEL_PREFIX + self.cfg.weights.split("mmdet://")[-1]
                )
            load_checkpoint(self, self.cfg.weights, revise_keys=REV_KEYS)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMTwoStageDetector training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.backbone(inputs)
        rpn_losses, proposals = self.rpn_head(inputs, features, targets)
        roi_losses, _ = self.roi_head(inputs, proposals, features, targets)
        return {**rpn_losses, **roi_losses}

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMTwoStageDetector testing!"
        inputs = batch_inputs[0]

        features = self.backbone(inputs)
        proposals = self.rpn_head(inputs, features)
        detections, segmentations = self.roi_head(inputs, proposals, features)

        outputs: Dict[str, List[TLabelInstance]] = dict(detect=detections)  # type: ignore # pylint: disable=line-too-long
        if self.with_mask:
            assert segmentations is not None
            outputs["ins_seg"] = segmentations

        return predictions_to_scalabel(
            inputs, outputs, self.cat_mapping, self.cfg.clip_bboxes_to_image
        )

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Detector feature extraction stage.

        Return backbone output features.
        """
        feats = self.backbone(inputs)
        assert isinstance(feats, dict)
        return feats

    def _proposals_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        """Train stage proposal generation."""
        return self.rpn_head(inputs, features, targets)

    def _proposals_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[Boxes2D]:
        """Test stage proposal generation."""
        return self.rpn_head(inputs, features)

    def _detections_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Train stage detections generation."""
        return self.roi_head(inputs, proposals, features, targets)

    def _detections_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Test stage detections generation."""
        return self.roi_head(inputs, proposals, features)


class MMOneStageDetectorConfig(BaseDetectorConfig):
    """Config for mmdetection one-stage models."""

    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    backbone_output_names: Optional[List[str]]
    weights: Optional[str]
    backbone: Optional[BaseBackboneConfig]
    bbox_head: Optional[BaseDenseHeadConfig]


class MMOneStageDetector(BaseOneStageDetector):
    """mmdetection one-stage detector wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMTwoStageDetector requires both mmcv and mmdet to be installed!"
        super().__init__(cfg)
        self.cfg: MMOneStageDetectorConfig = MMOneStageDetectorConfig(
            **cfg.dict()
        )
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.mm_cfg = get_mmdet_config(self.cfg)
        if self.cfg.backbone is None:
            self.cfg.backbone = MMDetBackboneConfig(
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
        self.backbone = build_backbone(self.cfg.backbone)

        if self.cfg.bbox_head is None:
            bbox_cfg = self.mm_cfg["bbox_head"]
            if "train_cfg" in self.mm_cfg:
                bbox_train_cfg = self.mm_cfg["train_cfg"]
            else:  # pragma: no cover
                bbox_train_cfg = None
            bbox_cfg.update(
                train_cfg=bbox_train_cfg,
                test_cfg=self.mm_cfg["test_cfg"],
            )
            self.cfg.bbox_head = MMDetDenseHeadConfig(
                type="MMDetDenseHead",
                mm_cfg=bbox_cfg,
                category_mapping=self.cfg.category_mapping,
            )
        self.bbox_head: BaseDenseHead[
            List[Boxes2D], List[Boxes2D]
        ] = build_dense_head(self.cfg.bbox_head)

        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmdet://"):
                self.cfg.weights = (
                    MMDET_MODEL_PREFIX + self.cfg.weights.split("mmdet://")[-1]
                )
            load_checkpoint(self, self.cfg.weights, revise_keys=REV_KEYS)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMOneStageDetector training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.backbone(inputs)
        bbox_losses, _ = self.bbox_head(inputs, features, targets)
        return {**bbox_losses}

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMOneStageDetector testing!"
        inputs = batch_inputs[0]
        features = self.backbone(inputs)
        detections = self.bbox_head(inputs, features)
        outputs = dict(detect=detections)
        return predictions_to_scalabel(
            inputs, outputs, self.cat_mapping, self.cfg.clip_bboxes_to_image
        )

    def extract_features(
        self, inputs: InputSample
    ) -> FeatureMaps:  # pragma: no cover
        """Detector feature extraction stage.

        Return backbone output features.
        """
        feats = self.backbone(inputs)
        assert isinstance(feats, dict)
        return feats

    def _detections_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[Boxes2D]]]:  # pragma: no cover
        """Train stage detections generation."""
        return self.bbox_head(inputs, features, targets)

    def _detections_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[Boxes2D]:  # pragma: no cover
        """Test stage detections generation."""
        return self.bbox_head(inputs, features)


def get_mmdet_config(
    config: Union[MMTwoStageDetectorConfig, MMOneStageDetectorConfig]
) -> MMConfig:  # pragma: no cover
    """Convert a Detector config to a mmdet readable config."""
    cfg = load_config(config.model_base)

    # convert detect attributes
    if (
        hasattr(config, "category_mapping")
        and config.category_mapping is not None
    ):
        if "bbox_head" in cfg:  # pragma: no cover
            cfg["bbox_head"]["num_classes"] = len(config.category_mapping)
        if "roi_head" in cfg:
            cfg["roi_head"]["bbox_head"]["num_classes"] = len(
                config.category_mapping
            )
            if "mask_head" in cfg["roi_head"]:
                cfg["roi_head"]["mask_head"]["num_classes"] = len(
                    config.category_mapping
                )

    if config.model_kwargs:
        add_keyword_args(config, cfg)
    return cfg
