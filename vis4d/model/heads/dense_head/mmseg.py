"""mmsegmentation decode head wrapper."""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from vis4d.common.utils.imports import MMCV_AVAILABLE, MMSEG_AVAILABLE

if MMCV_AVAILABLE:
    from mmcv.utils import ConfigDict


if MMSEG_AVAILABLE:
    from mmseg.models import build_head
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from vis4d.model.utils import (
    _parse_losses,
    get_img_metas,
    load_config,
    results_from_mmseg,
    targets_to_mmseg,
)
from vis4d.struct import (
    DictStrAny,
    FeatureMaps,
    InputSample,
    Losses,
    SemanticMasks,
)

from .base import SegDenseHead


class MMSegDecodeHead(SegDenseHead):
    """mmsegmentation decode head wrapper."""

    def __init__(
        self, mm_cfg: Union[DictStrAny, str], category_mapping: Dict[str, int]
    ):
        """Init."""
        assert (
            MMSEG_AVAILABLE and MMCV_AVAILABLE
        ), "MMSegDecodeHead requires both mmcv and mmseg to be installed!"
        super().__init__(category_mapping)
        mm_dict = (
            mm_cfg
            if isinstance(mm_cfg, dict)
            else load_config(mm_cfg, "decode_head")
        )
        self.train_cfg = mm_dict.pop("train_cfg", None)
        self.test_cfg = mm_dict.pop("test_cfg", None)
        self.mm_decode_head = build_head(ConfigDict(**mm_dict))
        assert isinstance(self.mm_decode_head, BaseDecodeHead)
        self.mm_decode_head.init_weights()
        self.mm_decode_head.train()
        assert category_mapping is not None
        self.cat_mapping = {v: k for k, v in category_mapping.items()}

    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets,
    ) -> Tuple[Losses, Optional[torch.Tensor]]:
        """Forward pass during training stage."""
        image_metas = get_img_metas(inputs.images)
        gt_masks = targets_to_mmseg(inputs.images, inputs.targets)
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
