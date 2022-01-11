"""mmdetection backbone wrapper."""
from typing import Optional

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
    weights: Optional[str]


class MMDetBackbone(BaseBackbone):
    """mmdetection backbone wrapper."""

    def __init__(self, cfg: BaseBackboneConfig):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetBackbone requires both mmcv and mmdet to be installed!"
        super().__init__(cfg)
        self.cfg: MMDetBackboneConfig = MMDetBackboneConfig(**cfg.dict())
        self.mm_backbone = build_backbone(self.cfg.mm_cfg)
        assert isinstance(self.mm_backbone, BaseModule)
        self.mm_backbone.init_weights()
        self.mm_backbone.train()

        self.neck: Optional[BaseNeck] = None
        if self.cfg.neck is not None:
            self.neck = build_neck(self.cfg.neck)

        if self.cfg.weights is not None:  # pragma: no cover
            if self.cfg.weights.startswith("mmdet://"):
                self.cfg.weights = (
                    MMDET_MODEL_PREFIX + self.cfg.weights.split("mmdet://")[-1]
                )
            load_checkpoint(self.mm_backbone, self.cfg.weights)

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
        backbone_outs = self.get_outputs(outs)
        if self.neck is not None:
            return self.neck(backbone_outs)
        return backbone_outs  # pragma: no cover
