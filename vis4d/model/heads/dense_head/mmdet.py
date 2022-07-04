"""mmdetection dense head wrapper."""
from typing import Dict, List, Optional, Tuple, Union

from vis4d.model.utils import (
    _parse_losses,
    get_img_metas,
    load_config,
    proposals_from_mmdet,
    targets_to_mmdet,
)
from vis4d.struct import (
    Boxes2D,
    DictStrAny,
    FeatureMaps,
    InputSample,
    LabelInstances,
    Losses,
)

from .base import DetDenseHead
from vis4d.common.utils.imports import MMCV_AVAILABLE, MMDET_AVAILABLE
if MMCV_AVAILABLE:
    from mmcv.utils import ConfigDict

if MMDET_AVAILABLE:
    from mmdet.models import build_head
    from mmdet.models.dense_heads.base_dense_head import (
        BaseDenseHead as MMBaseDenseHead,
    )


class MMDetDenseHead(DetDenseHead):
    """mmdetection dense head wrapper."""

    def __init__(
        self, mm_cfg: Union[DictStrAny, str], category_mapping: Dict[str, int]
    ) -> None:
        """Init."""
        assert (
            MMDET_AVAILABLE and MMCV_AVAILABLE
        ), "MMDetDenseHead requires both mmcv and mmdet to be installed!"
        super().__init__(category_mapping)
        mm_dict = (
            mm_cfg
            if isinstance(mm_cfg, dict)
            else load_config(mm_cfg, "dense_head")
        )
        self.mm_dense_head = build_head(ConfigDict(**mm_dict))
        assert isinstance(self.mm_dense_head, MMBaseDenseHead)
        self.mm_dense_head.init_weights()
        self.mm_dense_head.train()
        self.cat_mapping = {v: k for k, v in category_mapping.items()}
        self.proposal_cfg = self.mm_dense_head.train_cfg.pop(
            "rpn_proposal", self.mm_dense_head.test_cfg
        )

    def forward_train(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
        targets: LabelInstances,
    ) -> Tuple[Losses, List[Boxes2D]]:
        """Forward pass during training stage."""
        assert features is not None, "MMDetDenseHead requires features"
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        gt_bboxes, gt_labels, _ = targets_to_mmdet(targets)

        rpn_losses, proposals = self.mm_dense_head.forward_train(
            feat_list,
            img_metas,
            gt_bboxes,
            gt_labels=gt_labels,
            proposal_cfg=self.proposal_cfg,
        )
        return _parse_losses(rpn_losses), proposals_from_mmdet(proposals)

    def forward_test(
        self, inputs: InputSample, features: Optional[FeatureMaps]
    ) -> List[Boxes2D]:
        """Forward pass during testing stage."""
        assert features is not None, "MMDetDenseHead requires features"
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        proposals = self.mm_dense_head.simple_test(feat_list, img_metas)
        return proposals_from_mmdet(proposals)


class MMDetRPNHead(MMDetDenseHead):
    """mmdetection RPN head wrapper."""

    def forward_train(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
        targets: LabelInstances,
    ) -> Tuple[Losses, List[Boxes2D]]:
        """Forward pass during training stage."""
        assert features is not None, "MMDetRPNHead requires features"
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        gt_bboxes, _, _ = targets_to_mmdet(targets)

        rpn_losses, proposals = self.mm_dense_head.forward_train(
            feat_list,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            proposal_cfg=self.proposal_cfg,
        )
        return _parse_losses(rpn_losses), proposals_from_mmdet(proposals)
