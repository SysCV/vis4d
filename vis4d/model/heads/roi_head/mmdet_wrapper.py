"""mmdetection roi head wrapper."""
from typing import List, Optional, Sequence, Tuple

from vis4d.common.bbox.samplers import SamplingResult
from vis4d.model.mmdet_utils import (
    _parse_losses,
    detections_from_mmdet,
    get_img_metas,
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

    mm_cfg: DictStrAny


class MMDetRoIHead(BaseRoIHead[Tuple[Boxes2D, Optional[InstanceMasks]]]):
    """mmdetection roi head wrapper."""

    def __init__(self, cfg: BaseRoIHeadConfig) -> None:
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetRoIHead requires both mmcv and mmdet to be installed!"
        super().__init__()
        self.cfg: MMDetRoIHeadConfig = MMDetRoIHeadConfig(**cfg.dict())
        self.mm_roi_head = build_head(ConfigDict(**self.cfg.mm_cfg))
        assert isinstance(self.mm_roi_head, MMBaseRoIHead)
        self.mm_roi_head.init_weights()
        self.mm_roi_head.train()
        self.with_mask = self.mm_roi_head.with_mask

    def forward_train(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[FeatureMaps],
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
        features: Optional[FeatureMaps],
    ) -> Sequence[Tuple[Boxes2D, Optional[InstanceMasks]]]:
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
        else:
            segmentations = [None for _ in range(len(detections))]

        return list(zip(detections, segmentations))