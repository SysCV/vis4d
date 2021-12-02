"""mmdetection backbone wrapper."""
from typing import List, Optional, Tuple

import torch

try:
    from mmcv.runner import BaseModule
    from mmcv.runner.checkpoint import load_checkpoint

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmdet.models import build_backbone

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False

from vis4d.struct import DictStrAny, FeatureMaps, InputSample

from .base import BaseBackbone, BaseBackboneConfig
from .neck import BaseNeck, build_neck

MMDET_MODEL_PREFIX = "https://download.openmmlab.com/mmdetection/v2.0/"


class MMDetBackboneConfig(BaseBackboneConfig):
    """Config for mmdet backbones."""

    mm_cfg: DictStrAny
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    output_names: Optional[List[str]]
    weights: Optional[str]


class MMDetBackbone(BaseBackbone):
    """mmdetection backbone wrapper."""

    def __init__(self, cfg: BaseBackboneConfig):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetBackbone requires both mmcv and mmdet to be installed!"
        super().__init__()
        self.cfg: MMDetBackboneConfig = MMDetBackboneConfig(**cfg.dict())
        self.mm_backbone = build_backbone(self.cfg.mm_cfg)
        assert isinstance(self.mm_backbone, BaseModule)
        self.mm_backbone.init_weights()
        self.mm_backbone.train()

        self.neck: Optional[BaseNeck] = None
        if self.cfg.neck is not None:
            self.neck = build_neck(self.cfg.neck)

        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmdet://"):
                self.cfg.weights = (
                    MMDET_MODEL_PREFIX + self.cfg.weights.split("mmdet://")[-1]
                )
            load_checkpoint(self.mm_backbone, self.cfg.weights)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.cfg.pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.cfg.pixel_std).view(-1, 1, 1), False
        )

    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images."""
        inputs.images.tensor = (
            inputs.images.tensor - self.pixel_mean
        ) / self.pixel_std
        return inputs

    def forward(self, inputs: InputSample) -> FeatureMaps:  # type: ignore
        """Backbone forward.

        Args:
            inputs: Model Inputs, batched.

        Returns:
            FeatureMaps: Dictionary of output feature maps.
        """
        inputs = self.preprocess_inputs(inputs)
        outs = self.mm_backbone(inputs.images.tensor)
        if self.cfg.output_names is None:  # pragma: no cover
            backbone_outs = {f"out{i}": v for i, v in enumerate(outs)}
        else:
            backbone_outs = dict(zip(self.cfg.output_names, outs))
        if self.neck is not None:
            return self.neck(backbone_outs)
        return backbone_outs
