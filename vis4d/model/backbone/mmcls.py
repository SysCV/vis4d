"""mmclassification backbone wrapper."""
from typing import Optional, Union

try:
    from mmcv.runner import BaseModule
    from mmcv.runner.checkpoint import load_checkpoint
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmcls.models import build_backbone

    MMCLS_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCLS_INSTALLED = False

from vis4d.struct import ArgsType, DictStrAny, FeatureMaps, InputSample

from ..mm_utils import load_config
from .base import BaseBackbone

MMCLS_MODEL_PREFIX = "https://download.openmmlab.com/mmdetection/v2.0/"


class MMClsBackbone(BaseBackbone):
    """mmclassification backbone wrapper."""

    def __init__(
        self,
        mm_cfg: Union[DictStrAny, str],
        *args: ArgsType,
        weights: Optional[str] = None,
        **kwargs: ArgsType
    ):
        """Init."""
        assert (
            MMCLS_INSTALLED and MMCV_INSTALLED
        ), "MMClsBackbone requires both mmcv and mmcls to be installed!"
        super().__init__(*args, **kwargs)
        mm_dict = (
            mm_cfg
            if isinstance(mm_cfg, dict)
            else load_config(mm_cfg, "backbone")
        )
        self.mm_backbone = build_backbone(ConfigDict(**mm_dict))
        assert isinstance(self.mm_backbone, BaseModule)
        self.mm_backbone.init_weights()
        self.mm_backbone.train()

        if weights is not None:  # pragma: no cover
            if weights.startswith("mmcls://"):
                weights = MMCLS_MODEL_PREFIX + weights.split("mmcls://")[-1]
            load_checkpoint(self.mm_backbone, weights)

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
