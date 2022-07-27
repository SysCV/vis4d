"""Dense Head interface for Vis4D."""
import abc
from typing import Dict, Optional, Tuple, Union

from torch import nn

from vis4d.struct import (
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    TTestReturn,
    TTrainReturn,
)


class BaseDenseHead(nn.Module):
    """Base Dense head class."""

    def __init__(
        self, category_mapping: Optional[Dict[str, int]] = None
    ) -> None:
        """Init."""
        super().__init__()
        self.category_mapping = category_mapping

    def forward(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: Optional[LabelInstances] = None,
    ) -> Union[Tuple[LossesType, TTrainReturn], TTestReturn]:
        """Base Dense head forward.

        Args:
            inputs: Model Inputs, batched.
            features: Input feature maps.
            targets: Container with targets, e.g. Boxes2D / 3D, Masks, ...

        Returns:
            [LossesType, TTrainReturn] / TTestReturn: In train mode, return
            losses and intermediate outputs. In test mode, return predictions.
        """
        if targets is not None:
            return self.forward_train(inputs, features, targets)
        return self.forward_test(inputs, features)

    @abc.abstractmethod
    def forward_train(
        self,
        inputs: InputSample,
        features: FeatureMaps,
        targets: LabelInstances,
    ) -> Tuple[LossesType, TTrainReturn]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            targets: Targets corresponding to InputSamples.

        Returns:
            Tuple[LossesType, TTrainReturn]: Tuple of:
             (dict of scalar loss tensors, predictions / other outputs)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward_test(
        self, inputs: InputSample, features: FeatureMaps
    ) -> TTestReturn:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.

        Returns:
            TTestReturn: Prediction output.
        """
        raise NotImplementedError
