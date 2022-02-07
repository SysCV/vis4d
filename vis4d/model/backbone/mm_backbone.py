"""mmdetection backbone wrapper."""
from typing import Optional, Union

try:
    from mmcv.runner import BaseModule
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmdet.models import build_backbone as build_mmdet_backbone

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False

try:
    from mmseg.models import build_backbone as build_mmseg_backbone

    MMSEG_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMSEG_INSTALLED = False

try:
    from mmcls.models import build_backbone as build_mmcls_backbone

    MMCLS_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCLS_INSTALLED = False

from vis4d.struct import ArgsType, DictStrAny, FeatureMaps, InputSample, Images

from ..mm_utils import load_config, load_model_checkpoint
from .base import BaseBackbone


class MMDetBackbone(BaseBackbone):
    """mmdetection backbone wrapper."""

    def __init__(
        self,
        mm_cfg: Union[DictStrAny, str],
        *args: ArgsType,
        weights: Optional[str] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetBackbone requires both mmcv and mmdet to be installed!"
        super().__init__(*args, **kwargs)
        mm_dict = (
            mm_cfg
            if isinstance(mm_cfg, dict)
            else load_config(mm_cfg, "backbone")
        )
        self.mm_backbone = self.build_mm_backbone(ConfigDict(**mm_dict))
        assert isinstance(self.mm_backbone, BaseModule)
        self.mm_backbone.init_weights()
        self.mm_backbone.train()

        if weights is not None:  # pragma: no cover
            load_model_checkpoint(self.mm_backbone, weights)

    @staticmethod
    def build_mm_backbone(cfg: ConfigDict) -> BaseModule:
        """Build MM backbone with config."""
        return build_mmdet_backbone(cfg)

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


class MMSegBackbone(MMDetBackbone):
    """mmsegmentation backbone wrapper."""

    def __init__(
        self,
        mm_cfg: Union[DictStrAny, str],
        *args: ArgsType,
        weights: Optional[str] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMSEG_INSTALLED and MMCV_INSTALLED
        ), "MMSegBackbone requires both mmcv and mmseg to be installed!"
        super().__init__(mm_cfg, *args, weights=weights, **kwargs)

    @staticmethod
    def build_mm_backbone(cfg: ConfigDict) -> BaseModule:
        """Build MM backbone with config."""
        return build_mmseg_backbone(cfg)

    def preprocess_inputs(self, inputs: InputSample) -> InputSample:
        """Normalize the input images, pad masks."""
        if not self.training:
            # no padding during inference to match MMSegmentation
            Images.stride = 1
        super().preprocess_inputs(inputs)
        return inputs


class MMClsBackbone(MMDetBackbone):
    """mmclassification backbone wrapper."""

    def __init__(
        self,
        mm_cfg: Union[DictStrAny, str],
        *args: ArgsType,
        weights: Optional[str] = None,
        **kwargs: ArgsType,
    ):
        """Init."""
        assert (
            MMCLS_INSTALLED and MMCV_INSTALLED
        ), "MMClsBackbone requires both mmcv and mmcls to be installed!"
        super().__init__(mm_cfg, *args, weights=weights, **kwargs)

    @staticmethod
    def build_mm_backbone(cfg: ConfigDict) -> BaseModule:
        """Build MM backbone with config."""
        return build_mmcls_backbone(cfg)
