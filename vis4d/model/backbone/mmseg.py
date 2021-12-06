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
    from mmseg.models import build_backbone

    MMSEG_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMSEG_INSTALLED = False

from vis4d.struct import (
    DictStrAny,
    FeatureMaps,
    Images,
    InputSample,
    SemanticMasks,
)

from .base import BaseBackbone, BaseBackboneConfig
from .neck import BaseNeck, build_neck

MMSEG_MODEL_PREFIX = "https://download.openmmlab.com/mmsegmentation/v0.5/"


class MMSegBackboneConfig(BaseBackboneConfig):
    """Config for mmseg backbones."""

    mm_cfg: DictStrAny
    pixel_mean: Tuple[float, float, float]
    pixel_std: Tuple[float, float, float]
    output_names: Optional[List[str]]
    weights: Optional[str]


class MMSegBackbone(BaseBackbone):
    """mmsegmentation backbone wrapper."""

    def __init__(self, cfg: BaseBackboneConfig):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMSegBackbone requires both mmcv and mmseg to be installed!"
        super().__init__()
        self.cfg: MMSegBackboneConfig = MMSegBackboneConfig(**cfg.dict())
        self.mm_backbone = build_backbone(self.cfg.mm_cfg)
        assert isinstance(self.mm_backbone, BaseModule)
        self.mm_backbone.init_weights()
        self.mm_backbone.train()

        self.neck: Optional[BaseNeck] = None
        if self.cfg.neck is not None:
            self.neck = build_neck(self.cfg.neck)

        if self.cfg.weights is not None:
            if self.cfg.weights.startswith("mmseg://"):
                self.cfg.weights = (
                    MMSEG_MODEL_PREFIX + self.cfg.weights.split("mmseg://")[-1]
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
        """Normalize the input images, pad masks."""
        if not self.training:
            # no padding during inference to match MMSegmentation
            Images.stride = 1
        inputs.images.tensor = (
            inputs.images.tensor - self.pixel_mean
        ) / self.pixel_std
        if self.training and len(inputs.targets.semantic_masks) > 1:
            # pad masks to same size for batching
            inputs.targets.semantic_masks = SemanticMasks.pad(
                inputs.targets.semantic_masks,
                inputs.images.tensor.shape[-2:][::-1],
            )
        return inputs

    def __call__(  # type: ignore[override]
        self, inputs: InputSample
    ) -> FeatureMaps:
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
