"""Multi image classifier."""
from typing import List, Optional, Tuple, Union

from scalabel.label.typing import Frame

from vis4d.common.module import build_module
from vis4d.struct import (
    ArgsType,
    FeatureMaps,
    ImageTags,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    ModuleCfg,
)

from ..backbone import BaseBackbone
from ..heads.dense_head import ClsDenseHead, MultiClsHead
from ..utils import postprocess_predictions, predictions_to_scalabel
from .base import BaseClassifier


class MultiImageClassifier(BaseClassifier):
    """Multi image classifier."""

    def __init__(
        self,
        backbone: Union[BaseBackbone, ModuleCfg],
        head: Union[List[ClsDenseHead], List[ModuleCfg]],
        *args: ArgsType,
        **kwargs: ArgsType
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        if isinstance(backbone, dict):
            self.backbone: BaseBackbone = build_module(
                backbone, bound=BaseBackbone
            )
        else:  # pragma: no cover
            self.backbone = backbone

        self.head = MultiClsHead(head)

    def _multi_label_postprocess(
        self, inputs: InputSample, tags: List[ImageTags]
    ) -> ModelOutput:
        """Postprocess multi-label outputs."""
        outs = [Frame(name="", attributes={}) for _ in range(len(tags))]
        for i, cat_map in enumerate(self.head.category_mappings):
            outputs = dict(tagging=[tag[i] for tag in tags])
            postprocess_predictions(inputs, outputs)
            preds = predictions_to_scalabel(outputs, cat_map)
            for out, pred in zip(outs, preds["tagging"]):
                assert out.attributes is not None
                assert pred.attributes is not None
                out.attributes.update(pred.attributes)
        return {"tagging": outs}

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward pass during training stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MultiLabelClassifier training!"
        inputs, targets = batch_inputs[0], batch_inputs[0].targets
        assert targets is not None, "Training requires targets."
        features = self.extract_features(inputs)
        losses, _ = self.generate_classifications(inputs, features, targets)
        return losses

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage."""
        assert (
            len(batch_inputs) == 1
        ), "No reference views allowed in MultiLabelClassifier testing!"
        inputs = batch_inputs[0]
        features = self.extract_features(inputs)
        classifications = self.generate_classifications(inputs, features)
        assert classifications is not None
        return self._multi_label_postprocess(inputs, classifications)

    def extract_features(self, inputs: InputSample) -> FeatureMaps:
        """Classifier feature extraction stage."""
        return self.backbone(inputs)

    def _classifications_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, Optional[List[ImageTags]]]:
        """Train stage classifications generation."""
        return self.head(inputs, features, targets)

    def _classifications_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> List[ImageTags]:
        """Test stage classifications generation."""
        return self.head(inputs, features)
