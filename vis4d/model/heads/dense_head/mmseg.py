"""mmsegmentation decode head wrapper."""
import os
from typing import List, Optional, Sequence, Tuple, Union

import torch.nn.functional as F

try:
    from mmcv import Config as MMConfig
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmseg.models import build_head
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead

    MMSEG_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMSEG_INSTALLED = False

from vis4d.model.mmdet_utils import _parse_losses, get_img_metas
from vis4d.model.mmseg_utils import results_from_mmseg, targets_to_mmseg
from vis4d.struct import (
    DictStrAny,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    SemanticMasks,
)

from .base import BaseDenseHead, BaseDenseHeadConfig


class MMSegDecodeHeadConfig(BaseDenseHeadConfig):
    """Config for mmsegmentation decode heads."""

    mm_cfg: Union[DictStrAny, str]


class MMSegDecodeHead(
    BaseDenseHead[Optional[Sequence[SemanticMasks]], List[SemanticMasks]]
):
    """mmsegmentation decode head wrapper."""

    def __init__(self, cfg: BaseDenseHeadConfig):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMSegDecodeHead requires both mmcv and mmseg to be installed!"
        super().__init__()
        self.cfg: MMSegDecodeHeadConfig = MMSegDecodeHeadConfig(**cfg.dict())
        if isinstance(self.cfg.mm_cfg, dict):
            mm_cfg = self.cfg.mm_cfg
        else:
            # load from config
            assert os.path.exists(self.cfg.mm_cfg)
            mm_cfg = MMConfig.fromfile(self.cfg.mm_cfg)
            assert "decode_head" in mm_cfg
            mm_cfg = mm_cfg["decode_head"]
        self.train_cfg = mm_cfg.pop("train_cfg", None)
        self.test_cfg = mm_cfg.pop("test_cfg", None)
        self.mm_decode_head = build_head(ConfigDict(**mm_cfg))
        assert isinstance(self.mm_decode_head, BaseDecodeHead)
        self.mm_decode_head.init_weights()
        self.mm_decode_head.train()
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[Sequence[SemanticMasks]]]:
        """Forward pass during training stage."""
        image_metas = get_img_metas(inputs.images)
        gt_masks = targets_to_mmseg(inputs.targets)
        assert features is not None
        losses = self.mm_decode_head.forward_train(
            [features[k] for k in features.keys()],
            image_metas,
            gt_masks,
            self.train_cfg,
        )
        return _parse_losses(losses), None

    def forward_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[SemanticMasks]:
        """Forward pass during testing stage."""
        image_metas = get_img_metas(inputs.images)
        assert features is not None
        outs = self.mm_decode_head.forward_test(
            [features[k] for k in features.keys()], image_metas, self.test_cfg
        )
        outs = F.interpolate(
            outs,
            size=image_metas[0]["pad_shape"][:2],  # type: ignore
            mode="bilinear",
        )
        outs = outs.argmax(dim=1)
        return results_from_mmseg(outs, inputs.device)
