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
    LossesType,
)

from .base import BaseDenseBox2DHead

try:
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmdet.models import build_head
    from mmdet.models.dense_heads.base_dense_head import (
        BaseDenseHead as MMBaseDenseHead,
    )

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False


class MMDetDenseHead(BaseDenseBox2DHead):
    """mmdetection dense box2d head wrapper."""

    def __init__(
        self, mm_cfg: Union[DictStrAny, str], category_mapping: Dict[str, int]
    ) -> None:
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
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
        self.train_cfg = self.mm_dense_head.train_cfg.pop(
            "rpn_proposal", self.mm_dense_head.test_cfg
        )
        self.test_cfg = self.mm_dense_head.test_cfg

    def forward(
        self,
        features: Optional[FeatureMaps],
        targets: List[Boxes2D],
    ) -> Tuple[FeatureMaps, FeatureMaps]:
        """Forward pass during training stage."""
        cls_outs, box_outs = self.mm_dense_head(list(features.values()))
        return {k: v for k, v in zip(feat_list.keys(), cls_outs)}, {
            k: v for k, v in zip(feat_list.keys(), box_outs)
        }

    def postprocess(
        self,
        class_outs: FeatureMaps,
        regression_outs: FeatureMaps,
        images_shape: Tuple[int, int, int, int],
    ) -> List[Boxes2D]:
        """MMDet head postprocessing wrapper.

        Args:
            outputs (Tensor): Network outputs.

        Returns:
            List[Boxes2D]: Output boxes after postprocessing.
        """
        if self.training:
            cfg = self.train_cfg
        else:
            cfg = self.test_cfg

        boxes = self.mm_dense_head.get_bboxes(
            class_outs.values(),
            regression_outs.values(),
            get_img_metas(images_shape),
            cfg=cfg,
        )
        return proposals_from_mmdet(boxes)

    def loss(
        self,
        class_outs: FeatureMaps,
        regression_outs: FeatureMaps,
        targets: List[Boxes2D],
        images_shape: Tuple[int, int, int, int],
    ) -> LossesType:
        """MMDet head loss wrapper.

        Args:
            outputs: Network outputs.
            targets (List[Boxes2D]): Target 2D boxes.
            metadata (Dict): Dictionary of metadata needed for loss, e.g.
                image size, feature map strides, etc.
        Returns:
            LossesType: Dictionary of scalar loss tensors.
        """
        img_metas = get_img_metas(images_shape)
        gt_bboxes, gt_labels, _ = targets_to_mmdet(targets)
        self.mm_dense_head.loss(
            class_outs, regression_outs, gt_bboxes, gt_labels, img_metas
        )
