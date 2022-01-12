"""mmdetection backbone wrapper."""
from typing import Optional

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
    ArgsType,
    DictStrAny,
    FeatureMaps,
    Images,
    InputSample,
    SemanticMasks,
)

from .base import BaseBackbone

MMSEG_MODEL_PREFIX = "https://download.openmmlab.com/mmsegmentation/v0.5/"


class MMSegBackbone(BaseBackbone):
    """mmsegmentation backbone wrapper."""

    def __init__(
        self,
        mm_cfg: DictStrAny,
        *args: ArgsType,
        weights: Optional[str] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMSegBackbone requires both mmcv and mmseg to be installed!"
        super().__init__(*args, **kwargs)
        self.mm_backbone = build_backbone(mm_cfg)
        assert isinstance(self.mm_backbone, BaseModule)
        self.mm_backbone.init_weights()
        self.mm_backbone.train()

        if weights is not None:  # pragma: no cover
            if weights.startswith("mmseg://"):
                weights = MMSEG_MODEL_PREFIX + weights.split("mmseg://")[-1]
            load_checkpoint(self.mm_backbone, weights)

    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images, pad masks."""
        if not self.training:
            # no padding during inference to match MMSegmentation
            Images.stride = 1
        super().preprocess_inputs(inputs)
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
        backbone_outs = self.get_outputs(outs)
        if self.neck is not None:
            return self.neck(backbone_outs)
        return backbone_outs
