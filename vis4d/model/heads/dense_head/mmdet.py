"""mmdetection dense head wrapper."""
from typing import Dict, List, Optional, Tuple, Union

from vis4d.model.mmdet_utils import (
    _parse_losses,
    add_keyword_args,
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

from .base import BaseDenseHead, BaseDenseHeadConfig

try:
    from mmcv import Config as MMConfig
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


class MMDetDenseHeadConfig(BaseDenseHeadConfig):
    """Config for mmdetection dense heads."""

    mm_cfg: DictStrAny
    dense_head_name: str = "rpn_head"
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]


class MMDetDenseHead(BaseDenseHead[List[Boxes2D], List[Boxes2D]]):
    """mmdetection dense head wrapper."""

    def __init__(self, cfg: BaseDenseHeadConfig) -> None:
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetDenseHead requires both mmcv and mmdet to be installed!"
        super().__init__()
        self.cfg: MMDetDenseHeadConfig = MMDetDenseHeadConfig(**cfg.dict())
        if isinstance(self.cfg.mm_cfg, dict):
            mm_cfg = self.cfg.mm_cfg
        else:
            # load from config
            mm_cfg = get_mmdet_config(self.cfg)
        self.mm_dense_head = build_head(ConfigDict(**mm_cfg))
        assert isinstance(self.mm_dense_head, MMBaseDenseHead)
        self.mm_dense_head.init_weights()
        self.mm_dense_head.train()
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
        targets: LabelInstances,
    ) -> Tuple[LossesType, List[Boxes2D]]:
        """Forward pass during training stage."""
        assert features is not None, "MMDetDenseHead requires features"
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        gt_bboxes, _, _ = targets_to_mmdet(targets)

        proposal_cfg = self.mm_dense_head.train_cfg.get(
            "rpn_proposal", self.mm_dense_head.test_cfg
        )
        rpn_losses, proposals = self.mm_dense_head.forward_train(
            feat_list,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            proposal_cfg=proposal_cfg,
        )
        return _parse_losses(rpn_losses), proposals_from_mmdet(proposals)

    def forward_test(
        self,
        inputs: InputSample,
        features: Optional[FeatureMaps],
    ) -> List[Boxes2D]:
        """Forward pass during testing stage."""
        assert features is not None, "MMDetDenseHead requires features"
        feat_list = list(features.values())
        img_metas = get_img_metas(inputs.images)
        proposals = self.mm_dense_head.simple_test(feat_list, img_metas)
        return proposals_from_mmdet(proposals)


def get_mmdet_config(config: MMDetDenseHeadConfig) -> MMConfig:
    """Convert a Dense Head config to a mmdet readable config."""
    assert isinstance(config.mm_cfg, str)
    cfg = load_config(config.mm_cfg)

    # convert decode head attributes
    head_name = config.dense_head_name
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
