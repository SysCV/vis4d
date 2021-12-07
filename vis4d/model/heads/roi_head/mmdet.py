"""mmdetection roi head wrapper."""

from typing import Dict, List, Optional, Tuple, Union

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.model.mmdet_utils import (
    _parse_losses,
    add_keyword_args,
    detections_from_mmdet,
    get_img_metas,
    load_config,
    proposals_to_mmdet,
    segmentations_from_mmdet,
    targets_to_mmdet,
)
from vis4d.struct import (
    Boxes2D,
    DictStrAny,
    FeatureMaps,
    InputSample,
    InstanceMasks,
    LabelInstances,
    LossesType,
)

from .base import BaseRoIHead, BaseRoIHeadConfig

try:
    from mmcv import Config as MMConfig
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmdet.models import build_head
    from mmdet.models.roi_heads import BaseRoIHead as MMBaseRoIHead

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False


class MMDetRoIHeadConfig(BaseRoIHeadConfig):
    """Config for mmdetection roi heads."""

    mm_cfg: Union[DictStrAny, str]
    roi_head_name: str = "roi_head"
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]


class MMDetRoIHead(
    BaseRoIHead[
        Optional[SamplingResult],
        Tuple[List[Boxes2D], Optional[List[InstanceMasks]]],
    ]
):
    """mmdetection roi head wrapper."""

    def __init__(self, cfg: BaseRoIHeadConfig) -> None:
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetRoIHead requires both mmcv and mmdet to be installed!"
        super().__init__()
        self.cfg: MMDetRoIHeadConfig = MMDetRoIHeadConfig(**cfg.dict())
        if isinstance(self.cfg.mm_cfg, dict):
            mm_cfg = self.cfg.mm_cfg
        else:  # pragma: no cover
            # load from config
            mm_cfg = get_mmdet_config(self.cfg)
        self.mm_roi_head = build_head(ConfigDict(**mm_cfg))
        assert isinstance(self.mm_roi_head, MMBaseRoIHead)
        self.mm_roi_head.init_weights()
        self.mm_roi_head.train()
        self.with_mask = self.mm_roi_head.with_mask
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[SamplingResult]]:
        """Forward pass during training stage."""
        assert (
            boxes is not None
        ), "Generating detections with MMDetRoIHead requires proposals."
        assert features is not None, "MMDetRoIHead requires features"
        proposal_list = proposals_to_mmdet(boxes)
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)

        gt_bboxes, gt_labels, gt_masks = targets_to_mmdet(targets)
        detect_losses = self.mm_roi_head.forward_train(
            feat_list,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_masks=gt_masks,
        )
        return _parse_losses(detect_losses), None

    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: FeatureMaps,
    ) -> Tuple[List[Boxes2D], Optional[List[InstanceMasks]]]:
        """Forward pass during testing stage."""
        assert (
            boxes is not None
        ), "Generating detections with MMDetRoIHead requires proposals."
        assert features is not None, "MMDetRoIHead requires features"
        proposal_list = proposals_to_mmdet(boxes)
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)

        bboxes, labels = self.mm_roi_head.simple_test_bboxes(
            feat_list,
            img_metas,
            proposal_list,
            self.mm_roi_head.test_cfg,
        )
        detections = detections_from_mmdet(bboxes, labels)
        segmentations: Optional[List[InstanceMasks]] = None
        if self.with_mask:  # pragma: no cover
            masks = self.mm_roi_head.simple_test_mask(
                feat_list,
                img_metas,
                bboxes,
                labels,
            )
            segmentations = segmentations_from_mmdet(
                masks, detections, inputs.device
            )

        return detections, segmentations


def get_mmdet_config(
    config: MMDetRoIHeadConfig,
) -> MMConfig:  # pragma: no cover
    """Convert a RoI Head config to a mmdet readable config."""
    assert isinstance(config.mm_cfg, str)
    cfg = load_config(config.mm_cfg)

    # convert decode head attributes
    head_name = config.roi_head_name
    assert head_name in cfg
    if "num_classes" in cfg[head_name]:
        assert config.category_mapping is not None
        cfg[head_name]["num_classes"] = len(config.category_mapping)
    if "train_cfg" in cfg and head_name in cfg["train_cfg"]:
        cfg[head_name]["train_cfg"] = cfg["train_cfg"][head_name]
    if "test_cfg" in cfg and head_name in cfg["test_cfg"]:
        cfg[head_name]["test_cfg"] = cfg["test_cfg"][head_name]
    cfg = cfg[head_name]

    if config.model_kwargs:
        add_keyword_args(config, cfg)
    return cfg
