"""mmclassification neck wrapper."""
from typing import List, Optional, Union

from torch import nn

try:
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmcls.models import build_neck

    MMCLS_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCLS_INSTALLED = False

from vis4d.model.mm_utils import load_config
from vis4d.struct import DictStrAny, FeatureMaps

from .base import BaseNeck

MMCLS_MODEL_PREFIX = "https://download.openmmlab.com/mmclassification/v0/"


class MMClsNeck(BaseNeck):
    """mmclassification neck wrapper."""

    def __init__(
        self,
        mm_cfg: Union[DictStrAny, str],
        output_names: Optional[List[str]] = None,
    ):
        """Init."""
        assert (
            MMCLS_INSTALLED and MMCV_INSTALLED
        ), "MMClsNeck requires both mmcv and mmdet to be installed!"
        super().__init__()
        mm_dict = (
            mm_cfg if isinstance(mm_cfg, dict) else load_config(mm_cfg, "neck")
        )
        self.mm_neck = build_neck(ConfigDict(**mm_dict))
        self.output_names = output_names
        assert isinstance(self.mm_neck, nn.Module)
        self.mm_neck.init_weights()
        self.mm_neck.train()

    def __call__(  # type: ignore[override]
        self, inputs: FeatureMaps
    ) -> FeatureMaps:
        """Neck forward."""
        outs = self.mm_neck(tuple(inputs.values()))
        if self.output_names is None:
            return {f"out{i}": v for i, v in enumerate(outs)}
        return dict(zip(self.output_names, outs))  # pragma: no cover
