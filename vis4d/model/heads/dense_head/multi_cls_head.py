"""Multi classification head."""
import copy
from typing import List, Optional, Tuple, Union

import torch

from vis4d.common.module import build_module
from vis4d.model.mm_utils import _parse_losses
from vis4d.struct import (
    FeatureMaps,
    ImageTags,
    InputSample,
    LabelInstances,
    LossesType,
    ModuleCfg,
)

from .base import BaseDenseHead, ClsDenseHead


class MultiClsHead(ClsDenseHead):
    """Multi classification head."""

    def __init__(self, heads: Union[List[ClsDenseHead], List[ModuleCfg]]):
        """Init."""
        super().__init__()
        self.category_mappings = []
        for i, h in enumerate(heads):
            if isinstance(h, dict):
                assert h["category_mapping"] is not None
                assert h["tagging_attr"] is not None
                head: ClsDenseHead = build_module(h, bound=BaseDenseHead)
            else:
                head = h  # type: ignore
            self.add_module(str(i), head)
            assert head.category_mapping is not None
            self.category_mappings.append(
                {v: k for k, v in head.category_mapping.items()}
            )

    def __len__(self) -> int:
        """Return number of classification heads."""
        return len(self.category_mappings)

    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[torch.Tensor]]]:
        """Forward pass during training stage."""
        cls_losses = {}
        for i in range(len(self)):
            head_targets = copy.deepcopy(targets)
            head_targets.image_tags = [t[i] for t in head_targets.image_tags]
            cls_losses.update(
                _parse_losses(
                    getattr(self, str(i))(inputs, features, head_targets)[0]
                )
            )
        return cls_losses, None

    def forward_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[ImageTags]:
        """Forward pass during testing stage."""
        outs = [
            getattr(self, str(i))(inputs, features) for i in range(len(self))
        ]
        return [
            ImageTags.merge([t[i] for t in outs]) for i in range(len(outs[0]))
        ]
