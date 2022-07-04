"""mmdetection backbone wrapper."""
from typing import Optional, Union

from vis4d.common.utils.imports import MMCV_AVAILABLE, MMDET_AVAILABLE, MMSEG_AVAILABLE
if MMCV_AVAILABLE:
    from mmcv.runner import BaseModule
    from mmcv.utils import ConfigDict

if MMDET_AVAILABLE:
    from mmdet.models import build_backbone as build_mmdet_backbone


if MMSEG_AVAIABLE:
    from mmseg.models import build_backbone as build_mmseg_backbone

from vis4d.struct import ArgsType, DictStrAny, FeatureMaps, Images, InputSample

from ..utils import load_config, load_model_checkpoint
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
            MMDET_AVAILABLE and MMCV_AVAILABLE
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
            MMSEG_AVAILABLE and MMCV_AVAILABLE
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
