"""mmsegmentation decode head wrapper."""
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

try:
    from mmcv import Config as MMConfig

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmseg.models import build_head
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead

    MMSEG_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMSEG_INSTALLED = False


from vis4d.model.base import BaseModelConfig
from vis4d.model.detect.mmdet_utils import (
    _parse_losses,
    add_keyword_args,
    get_img_metas,
)
from vis4d.model.segment.mmseg_utils import (
    load_config_from_mmseg,
    results_from_mmseg,
    targets_to_mmseg,
)
from vis4d.struct import InputSample, LabelInstances, LossesType, SemanticMasks

from .base import BaseDenseHead, BaseDenseHeadConfig


class MMDecodeHeadConfig(BaseDenseHeadConfig):
    """Config for mmsegmentation decode heads."""

    name: str = "decode_head"
    model_base: str
    model_kwargs: Optional[Dict[str, Union[bool, float, str, List[float]]]]


def get_mmseg_config(config: MMDecodeHeadConfig) -> MMConfig:
    """Convert a DecodeHead config to a mmseg readable config."""
    if os.path.exists(config.model_base):  # pragma: no cover
        cfg = MMConfig.fromfile(config.model_base)
        if cfg.get("model"):
            cfg = cfg["model"]
    elif config.model_base.startswith("mmseg://"):
        ex = os.path.splitext(config.model_base)[1]
        cfg = MMConfig.fromstring(
            load_config_from_mmseg(config.model_base.split("mmseg://")[-1]), ex
        ).model
    else:
        raise FileNotFoundError(
            f"MMSegmentation config not found: {config.model_base}"
        )
    assert config.name in cfg, f"DecodeHead config not found: {config.name}"
    cfg = cfg[config.name]

    # convert decode head attributes
    assert config.category_mapping is not None
    cfg["num_classes"] = len(config.category_mapping)

    if config.model_kwargs:
        add_keyword_args(config, cfg)
    return cfg


class MMDecodeHead(BaseDenseHead[SemanticMasks]):
    """mmsegmentation decode head wrapper."""

    def __init__(self, cfg: BaseModelConfig):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMDecodeHead requires both mmcv and mmseg to be installed!"
        super().__init__()
        self.cfg: MMDecodeHeadConfig = MMDecodeHeadConfig(**cfg.dict())
        self.mm_cfg = get_mmseg_config(self.cfg)
        self.mm_decode_head = build_head(self.mm_cfg)
        assert isinstance(self.mm_decode_head, BaseDecodeHead)
        self.mm_decode_head.init_weights()
        self.mm_decode_head.train()
        assert self.cfg.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

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
        outs = F.interpolate(
            outs,
            size=image_metas[0]["pad_shape"][:2],  # type: ignore
            mode="bilinear",
        )
        outs = outs.argmax(dim=1)
        return results_from_mmseg(outs, image_metas, inputs.device)
