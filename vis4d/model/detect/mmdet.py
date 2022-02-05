"""mmdetection detector wrapper."""
from typing import Dict, List, Optional, Tuple, Union

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.module import build_module
from vis4d.struct import (
    ArgsType,
    Boxes2D,
    DictStrAny,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
    ModelOutput,
    ModuleCfg,
    TLabelInstance,
)

from ..backbone import BaseBackbone, MMDetBackbone
from ..backbone.neck import MMDetNeck
from ..heads.dense_head import (
    BaseDenseHead,
    DetDenseHead,
    MMDetDenseHead,
    MMDetRPNHead,
)
from ..heads.roi_head import BaseRoIHead, Det2DRoIHead, MMDetRoIHead
from ..mm_utils import add_keyword_args, load_config, load_model_checkpoint
from ..utils import postprocess_predictions, predictions_to_scalabel
from .base import BaseOneStageDetector, BaseTwoStageDetector

try:
    from mmcv import Config as MMConfig

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


class MMTwoStageDetector(BaseTwoStageDetector):
    """mmdetection two-stage detector wrapper."""

    def __init__(
        self,
        model_base: str,
        *args: ArgsType,
        pixel_mean: Optional[Tuple[float, float, float]] = None,
        pixel_std: Optional[Tuple[float, float, float]] = None,
        model_kwargs: Optional[DictStrAny] = None,
        backbone_output_names: Optional[List[str]] = None,
        weights: Optional[str] = None,
        backbone: Optional[Union[BaseBackbone, ModuleCfg]] = None,
        rpn_head: Optional[Union[DetDenseHead, ModuleCfg]] = None,
        roi_head: Optional[Union[Det2DRoIHead, ModuleCfg]] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMTwoStageDetector requires both mmcv and mmdet to be installed!"
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}
        self.mm_cfg = get_mmdet_config(
            model_base, model_kwargs, self.category_mapping
        )
        if pixel_mean is None or pixel_std is None:
            assert backbone is not None, (
                "If no custom backbone is defined, image "
                "normalization parameters must be specified!"
            )

        if backbone is None:
            self.backbone: BaseBackbone = MMDetBackbone(
                mm_cfg=self.mm_cfg["backbone"],
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                neck=MMDetNeck(
                    mm_cfg=self.mm_cfg["neck"],
                    output_names=backbone_output_names,
                ),
            )
        elif isinstance(backbone, dict):
            self.backbone = build_module(backbone, bound=BaseBackbone)
        else:  # pragma: no cover
            self.backbone = backbone

        if rpn_head is None:
            rpn_cfg = self.mm_cfg["rpn_head"]
            if (
                "train_cfg" in self.mm_cfg
                and "rpn" in self.mm_cfg["train_cfg"]
            ):
                rpn_train_cfg = self.mm_cfg["train_cfg"]["rpn"]
            else:  # pragma: no cover
                rpn_train_cfg = None
            if (
                "train_cfg" in self.mm_cfg
                and "rpn_proposal" in self.mm_cfg["train_cfg"]
            ):
                rpn_proposal_cfg = self.mm_cfg["train_cfg"]["rpn_proposal"]
                if rpn_train_cfg is not None:
                    rpn_train_cfg.update(rpn_proposal=rpn_proposal_cfg)
                else:  # pragma: no cover
                    rpn_train_cfg = rpn_proposal_cfg
            else:  # pragma: no cover
                rpn_train_cfg = None
            rpn_cfg.update(
                train_cfg=rpn_train_cfg,
                test_cfg=self.mm_cfg["test_cfg"]["rpn"],
            )
            self.rpn_head = MMDetRPNHead(
                mm_cfg=rpn_cfg, category_mapping=self.category_mapping
            )
        elif isinstance(rpn_head, dict):  # pragma: no cover
            self.rpn_head = build_module(rpn_head, bound=BaseDenseHead)
        else:  # pragma: no cover
            self.rpn_head = rpn_head

        if roi_head is None:
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
            self.roi_head = MMDetRoIHead(
                mm_cfg=roi_head_cfg,
                category_mapping=self.category_mapping,
            )
        elif isinstance(roi_head, dict):  # pragma: no cover
            self.roi_head = build_module(roi_head, bound=BaseRoIHead)
        else:  # pragma: no cover
            self.roi_head = roi_head

        self.with_mask = self.roi_head.with_mask
        if weights is not None:
            load_model_checkpoint(self, weights, REV_KEYS)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMTwoStageDetector training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.backbone(inputs)
        rpn_losses, proposals = self.rpn_head(inputs, features, targets)
        roi_losses, _ = self.roi_head(inputs, features, proposals, targets)
        return {**rpn_losses, **roi_losses}

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMTwoStageDetector testing!"
        inputs = batch_inputs[0]
        features = self.backbone(inputs)
        proposals = self.rpn_head(inputs, features)
        detections, segmentations = self.roi_head(inputs, features, proposals)
        outputs: Dict[str, List[TLabelInstance]] = dict(detect=detections)  # type: ignore # pylint: disable=line-too-long
        if self.with_mask:
            assert segmentations is not None
            outputs["ins_seg"] = segmentations

        postprocess_predictions(inputs, outputs, self.clip_bboxes_to_image)
        return predictions_to_scalabel(outputs, self.cat_mapping)

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
        return self.roi_head(inputs, features, proposals, targets)

    def _detections_test(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        proposals: List[Boxes2D],
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Test stage detections generation."""
        return self.roi_head(inputs, features, proposals)


class MMOneStageDetector(BaseOneStageDetector):
    """mmdetection one-stage detector wrapper."""

    def __init__(
        self,
        model_base: str,
        pixel_mean: Tuple[float, float, float],
        pixel_std: Tuple[float, float, float],
        *args: ArgsType,
        model_kwargs: Optional[DictStrAny] = None,
        backbone_output_names: Optional[List[str]] = None,
        weights: Optional[str] = None,
        backbone: Optional[Union[BaseBackbone, ModuleCfg]] = None,
        bbox_head: Optional[Union[DetDenseHead, ModuleCfg]] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMTwoStageDetector requires both mmcv and mmdet to be installed!"
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}
        self.mm_cfg = get_mmdet_config(
            model_base, model_kwargs, self.category_mapping
        )
        if backbone is None:
            self.backbone: BaseBackbone = MMDetBackbone(
                mm_cfg=self.mm_cfg["backbone"],
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                neck=MMDetNeck(
                    mm_cfg=self.mm_cfg["neck"],
                    output_names=backbone_output_names,
                ),
            )
        elif isinstance(backbone, dict):  # pragma: no cover
            self.backbone = build_module(backbone, bound=BaseBackbone)
        else:  # pragma: no cover
            self.backbone = backbone

        if bbox_head is None:
            bbox_cfg = self.mm_cfg["bbox_head"]
            if "train_cfg" in self.mm_cfg:
                bbox_train_cfg = self.mm_cfg["train_cfg"]
            else:  # pragma: no cover
                bbox_train_cfg = None
            bbox_cfg.update(
                train_cfg=bbox_train_cfg,
                test_cfg=self.mm_cfg["test_cfg"],
            )
            self.bbox_head: DetDenseHead = MMDetDenseHead(
                mm_cfg=bbox_cfg, category_mapping=self.category_mapping
            )
        elif isinstance(bbox_head, dict):  # pragma: no cover
            self.bbox_head = build_module(bbox_head, bound=BaseDenseHead)
        else:  # pragma: no cover
            self.bbox_head = bbox_head

        if weights is not None:
            load_model_checkpoint(self, weights, REV_KEYS)

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
        postprocess_predictions(inputs, outputs, self.clip_bboxes_to_image)
        return predictions_to_scalabel(outputs, self.cat_mapping)

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
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
    ) -> Tuple[LossesType, Optional[List[Boxes2D]]]:
        """Train stage detections generation."""
        return self.bbox_head(inputs, features, targets)

    def _detections_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[Boxes2D]:
        """Test stage detections generation."""
        return self.bbox_head(inputs, features)


def get_mmdet_config(
    model_base: str,
    model_kwargs: Optional[DictStrAny] = None,
    category_mapping: Optional[Dict[str, int]] = None,
) -> MMConfig:
    """Convert a Detector config to a mmdet readable config."""
    cfg = load_config(model_base)

    # convert detect attributes
    if category_mapping is not None:
        if "bbox_head" in cfg:  # pragma: no cover
            cfg["bbox_head"]["num_classes"] = len(category_mapping)
        if "roi_head" in cfg:
            cfg["roi_head"]["bbox_head"]["num_classes"] = len(category_mapping)
            if "mask_head" in cfg["roi_head"]:
                cfg["roi_head"]["mask_head"]["num_classes"] = len(
                    category_mapping
                )

    if model_kwargs is not None:
        add_keyword_args(model_kwargs, cfg)
    return cfg
