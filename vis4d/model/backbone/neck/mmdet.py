"""mmdetection neck wrapper."""
from typing import List, Optional

try:
    from mmcv.runner import BaseModule

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmdet.models import build_neck

    MMDET_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMDET_INSTALLED = False

from vis4d.struct import DictStrAny, FeatureMaps

from .base import BaseNeck

MMDET_MODEL_PREFIX = "https://download.openmmlab.com/mmdetection/v2.0/"


class MMDetNeck(BaseNeck):
    """mmdetection neck wrapper."""

    def __init__(
        self,
        mm_cfg: DictStrAny,
        output_names: Optional[List[str]] = None,
    ):
        """Init."""
        assert (
            MMDET_INSTALLED and MMCV_INSTALLED
        ), "MMDetNeck requires both mmcv and mmdet to be installed!"
        super().__init__()
        self.output_names = output_names
        self.mm_neck = build_neck(mm_cfg)
        assert isinstance(self.mm_neck, BaseModule)
        self.mm_neck.init_weights()
        self.mm_neck.train()

    def __call__(  # type: ignore[override]
        self,
        inputs: FeatureMaps,
    ) -> FeatureMaps:
        """Neck forward."""
        outs = self.mm_neck(list(inputs.values()))
        if self.output_names is None:  # pragma: no cover
            return {f"out{i}": v for i, v in enumerate(outs)}
        return dict(zip(self.output_names, outs))
