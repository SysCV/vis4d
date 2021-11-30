"""mmsegmentation decode head wrapper."""
from typing import Dict, List, Optional

import torch

try:
    from mmseg.models import build_head
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead

    MMSEG_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMSEG_INSTALLED = False


from vis4d.model.base import BaseModelConfig
from vis4d.model.detect.mmdet_utils import _parse_losses, get_img_metas
from vis4d.model.segment.mmseg_utils import targets_to_mmseg
from vis4d.struct import InputSample, LabelInstances, LossesType, SemanticMasks

from .base import BaseDenseHead
from .mmseg_utils import (
    MMDecodeHeadConfig,
    get_mmseg_config,
    results_from_mmseg,
)


class MMDecodeHead(BaseDenseHead[SemanticMasks]):
    """mmsegmentation decode head wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        assert MMSEG_INSTALLED, "MMDecodeHead requires mmseg to be installed!"
        super().__init__()
        self.cfg: MMDecodeHeadConfig = MMDecodeHeadConfig(**cfg.dict())
        self.mm_cfg = get_mmseg_config(self.cfg)
        self.mm_decode_head = build_head(self.mm_cfg)
        assert isinstance(self.mm_decode_head, BaseDecodeHead)
        self.mm_decode_head.init_weights()
        self.mm_decode_head.train()

    def forward_train(
        self,
        inputs: InputSample,
        features: Optional[Dict[str, torch.Tensor]],
        targets: LabelInstances,
    ) -> LossesType:
        """Forward pass during training stage."""
        image_metas = get_img_metas(inputs.images)
        gt_masks = targets_to_mmseg(inputs.targets)
        assert features is not None
        losses = self.mm_decode_head.forward_train(
            [features[k] for k in features.keys()], image_metas, gt_masks, None
        )
        return _parse_losses(losses)

    def forward_test(
        self,
        inputs: InputSample,
        features: Optional[Dict[str, torch.Tensor]],
    ) -> List[SemanticMasks]:
        """Forward pass during testing stage."""
        image_metas = get_img_metas(inputs.images)
        assert features is not None
        outs = self.mm_decode_head.forward_test(
            [features[k] for k in features.keys()], image_metas, None
        )
        return results_from_mmseg(outs, image_metas, inputs.device)
