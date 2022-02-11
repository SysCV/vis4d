"""mmclassification classifier wrapper."""
from typing import Dict, List, Optional, Tuple, Union

from vis4d.common.module import build_module
from vis4d.struct import (
    ArgsType,
    DictStrAny,
    FeatureMaps,
    ImageTags,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    ModuleCfg,
)

from ..backbone import BaseBackbone, MMClsBackbone
from ..backbone.neck import MMClsNeck
from ..heads.dense_head import ClsDenseHead, MMClsHead
from ..mm_utils import (
    _parse_losses,
    add_keyword_args,
    load_config,
    load_model_checkpoint,
)
from ..utils import postprocess_predictions, predictions_to_scalabel
from .base import BaseClassifier

try:
    from mmcv import Config as MMConfig

    MMCV_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCV_INSTALLED = False

try:
    MMCLS_INSTALLED = True
except (ImportError, NameError):  # pragma: no cover
    MMCLS_INSTALLED = False


REV_KEYS = [
    (r"^head\.", "head.mm_cls_head."),
    (r"^backbone\.", "backbone.mm_backbone."),
    (r"^neck\.", "backbone.neck.mm_neck."),
]


class MMImageClassifier(BaseClassifier):
    """mmclassification image classifier wrapper."""

    def __init__(
        self,
        model_base: str,
        tagging_attr: str,
        *args: ArgsType,
        pixel_mean: Optional[Tuple[float, float, float]] = None,
        pixel_std: Optional[Tuple[float, float, float]] = None,
        model_kwargs: Optional[DictStrAny] = None,
        backbone_output_names: Optional[List[str]] = None,
        weights: Optional[str] = None,
        backbone: Optional[Union[BaseBackbone, ModuleCfg]] = None,
        head: Optional[Union[ClsDenseHead, ModuleCfg]] = None,
        **kwargs: ArgsType
    ):
        """Init."""
        assert (
            MMCLS_INSTALLED and MMCV_INSTALLED
        ), "MMImageClassifier requires both mmcv and mmcls to be installed!"
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}
        self.mm_cfg = get_mmcls_config(
            model_base, model_kwargs, self.category_mapping
        )
        if pixel_mean is None or pixel_std is None:  # pragma: no cover
            assert backbone is not None, (
                "If no custom backbone is defined, image "
                "normalization parameters must be specified!"
            )

        if backbone is None:
            self.backbone: BaseBackbone = MMClsBackbone(
                mm_cfg=self.mm_cfg["backbone"],
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                neck=MMClsNeck(
                    mm_cfg=self.mm_cfg["neck"],
                    output_names=backbone_output_names,
                ),
            )
        elif isinstance(backbone, dict):  # pragma: no cover
            self.backbone = build_module(backbone, bound=BaseBackbone)
        else:  # pragma: no cover
            self.backbone = backbone

        if head is None:
            self.head = MMClsHead(
                mm_cfg=self.mm_cfg["head"],
                category_mapping=self.category_mapping,
                tagging_attr=tagging_attr,
            )
        elif isinstance(head, dict):  # pragma: no cover
            self.head = build_module(head, bound=ClsDenseHead)
        else:  # pragma: no cover
            self.head = head

        if weights is not None:
            load_model_checkpoint(self, weights, REV_KEYS)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMImageClassifier training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.extract_features(inputs)
        losses, _ = self.generate_classifications(inputs, features, targets)
        return losses

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MMImageClassifier testing!"
        inputs = batch_inputs[0]
        features = self.extract_features(inputs)
        classifications = self.generate_classifications(inputs, features)
        assert classifications is not None

        outputs = dict(tagging=classifications)
        postprocess_predictions(inputs, outputs)
        return predictions_to_scalabel(outputs, self.cat_mapping)

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Classifier feature extraction stage.

        Return backbone output features.
        """
        return self.backbone(inputs)

    def _classifications_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[ImageTags]]]:
        """Train stage classifications generation."""
        cls_losses, _ = self.head(inputs, features, targets)
        return _parse_losses(cls_losses), None

    def _classifications_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[ImageTags]:
        """Test stage classifications generation."""
        return self.head(inputs, features)


def get_mmcls_config(
    model_base: str,
    model_kwargs: Optional[DictStrAny] = None,
    category_mapping: Optional[Dict[str, int]] = None,
) -> MMConfig:
    """Convert a Classifier config to a mmcls readable config."""
    cfg = load_config(model_base)

    # convert classifier attributes
    assert category_mapping is not None
    cfg["head"]["num_classes"] = len(category_mapping)

    if model_kwargs:
        add_keyword_args(model_kwargs, cfg)
    return cfg
