"""mmclassification classification head wrapper."""
from typing import Dict, List, Optional, Tuple, Union

try:
    from mmcv.utils import ConfigDict

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    from mmcls.models import build_head
    from mmcls.models.heads.base_head import BaseHead

    MMCLS_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCLS_INSTALLED = False

from vis4d.model.mm_utils import (
    _parse_losses,
    load_config,
    results_from_mmcls,
    targets_to_mmcls,
)
from vis4d.struct import (
    DictStrAny,
    FeatureMaps,
    ImageTags,
    InputSample,
    LabelInstances,
    LossesType,
)

from .base import ClsDenseHead


class MMClsHead(ClsDenseHead):
    """mmsegmentation decode head wrapper."""

    def __init__(
        self,
        mm_cfg: Union[DictStrAny, str],
        category_mapping: Dict[str, int],
        tagging_attribute: str,
    ):
        """Init."""
        assert (
            MMCLS_INSTALLED and MMCV_INSTALLED
        ), "MMClsHead requires both mmcv and mmcls to be installed!"
        super().__init__(category_mapping)
        self.tagging_attribute = tagging_attribute
        mm_dict = (
            mm_cfg if isinstance(mm_cfg, dict) else load_config(mm_cfg, "head")
        )
        self.mm_cls_head = build_head(ConfigDict(**mm_dict))
        assert isinstance(self.mm_cls_head, BaseHead)
        self.mm_cls_head.init_weights()
        self.mm_cls_head.train()
        assert self.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}

    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[ImageTags]]]:
        """Forward pass during training stage."""
        gt_labels = targets_to_mmcls(targets)
        losses = self.mm_cls_head.forward_train(
            tuple(features[k] for k in features.keys()), gt_labels
        )
        return _parse_losses(losses, self.tagging_attribute), None

    def forward_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[ImageTags]:
        """Forward pass during testing stage."""
        outs = self.mm_cls_head.simple_test(
            tuple(features[k] for k in features.keys()), post_process=False
        )
        return results_from_mmcls(outs, self.tagging_attribute)
