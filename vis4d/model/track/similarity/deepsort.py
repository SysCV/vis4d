"""Similarity Head for the deepsort tracking network.
 Mostly taken and adapted from https://github.com/nwojke/cosine_metric_learning
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_warn
from torch import nn

from vis4d.common.bbox.poolers import BaseRoIPooler, MultiScaleRoIAlign
from vis4d.common.bbox.samplers import SamplingResult
from vis4d.common.layers import Conv2d, ResidualBlock
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
)

from .base import BaseSimilarityHead


class NormalizedLogitLayer(nn.Linear):
    """
    NormalizedLogitLayer last layer of Cosine Similarity Learning
    This is a linear layer with custom scale factor and normalized weights:

    f(x) = scale * normalize(W) * x

    Original Code (tensor flow) :
    (https://github.com/nwojke/cosine_metric_learning/blob
    /eda0daaa5462c61ac44553f2151070fd7e316cc8/nets/deep_sort
    /network_definition.py#L87)
    """

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            **kwargs
        )
        self.scale = nn.Parameter(torch.empty(1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight: nn.Parameter = nn.Parameter(
                F.normalize(self.weight, dim=1)
            )  # weights has dimension (out, in) -> normalize over in channels
            self.scale = nn.Parameter(F.softplus(self.scale))

        return self.scale * F.linear(input, self.weight, None)


class DeepSortSimilarityHead(BaseSimilarityHead):
    """Instance embedding head for deep sort model"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        proposal_pooler: Optional[BaseRoIPooler] = None,
        projection_dim: int = 128,
        num_fc_layers: int = 1,
        fc_out_dim: int = 128,
        max_boxes_num: int = 128,
        dropout_probability: float = 0.6,
        activation="ELU",
        norm: str = "BatchNorm2d",
        num_classes: int = 10,
    ) -> None:
        """
        Craetes a new similarity head that extracts features for each
        bounding box and image
        Args:
            proposal_pooler: BaseRoIPooler, The pooler that should be used for
                the detected bounding boxes. If none is specified, a default
                MultiScaleRoIAlign with [64x128] resolution will be used
            projection_dim: int,  Number of channels of the extracted features
                                  for each pooled bounding box before passing
                                  them into the MLP
            num_fc_layers: , int  How many fully convolutional layers to use
                            in the final MLP
            fc_out_dim:  int,   Dimensionality of the final feature encoding
            max_boxes_num: int,  Max number of boxes that can be used at each
                              step
            dropout_probability: float, Dropout probability for all dropout
            layers in this architecture
            activation:  str, Activation function (str) that should be used
            norm: str, Norm function (str) that should be used
            num_classes: int, Final number of classes. Only used during
            the training stage
        """
        super().__init__()
        self.proposal_pooler = proposal_pooler

        if proposal_pooler is not None:
            self.roi_pooler = proposal_pooler
        else:
            self.roi_pooler = MultiScaleRoIAlign(
                resolution=[64, 128], strides=[1], sampling_ratio=0
            )

        self.projection_dim = projection_dim
        self.num_fc_layers = num_fc_layers
        self.fc_out_dim = fc_out_dim
        self.dropout_probability = dropout_probability
        self.max_boxes_num = max_boxes_num
        self.activation = activation
        self.norm = norm

        # Build Model
        self.backbone, self.backbone_dims = self._build_backbone()
        self.fc_embedder = self._build_mlp_layers()
        self.num_classes = num_classes
        self.classifier_loss = nn.CrossEntropyLoss()
        self.classifier = NormalizedLogitLayer(fc_out_dim, self.num_classes)

    def _build_backbone(self) -> Tuple[nn.Sequential, Tuple[int, int, int]]:
        """
        Builds the backbone network that extracts feature maps
        Returns:
         torch.nn.Sequential: Backbone network
         Tuple[int,int.int]: final extracted dimensions [C,H,W]
        """
        return nn.Sequential(
            Conv2d(
                in_channels=3,
                out_channels=self.projection_dim // 4,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                norm=getattr(nn, self.norm)(
                    self.projection_dim // 4,
                )
                if self.norm is not None
                else None,
                activation=getattr(nn, self.activation)()
                if self.activation is not None
                else None,
            ),
            Conv2d(
                in_channels=self.projection_dim // 4,
                out_channels=self.projection_dim // 4,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                norm=getattr(nn, self.norm)(
                    self.projection_dim // 4,
                )
                if self.norm is not None
                else None,
                activation=getattr(nn, self.activation)(inplace=True)
                if self.activation is not None
                else None,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(
                self.projection_dim // 4,
                self.projection_dim // 4,
                activation_cfg=self.activation,
            ),
            ResidualBlock(
                self.projection_dim // 4,
                self.projection_dim // 2,
                stride=2,
                activation_cfg=self.activation,
            ),
            ResidualBlock(
                self.projection_dim // 2,
                self.projection_dim,
                stride=2,
                activation_cfg=self.activation,
            ),
        ), (
            self.projection_dim,
            self.roi_pooler.resolution[0] // (2**3),
            self.roi_pooler.resolution[1] // (2**3),
        )

    def _build_mlp_layers(
        self,
    ) -> torch.nn.Sequential:
        """
        Builds the final MLP extracting a 1D latent representation for each
        feature map obtained from the backbone network
        Returns:
          torch.nn.Sequential: MLP network
        """
        if self.num_fc_layers > 0:
            input_dim = np.prod(self.backbone_dims).item()
            return nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Dropout(p=self.dropout_probability),
                        nn.Linear(input_dim, self.fc_out_dim),
                        nn.BatchNorm1d(self.fc_out_dim),
                        getattr(nn, self.activation)(inplace=True),
                    )
                    for _ in range(self.num_fc_layers)
                ]
            )
        return nn.Sequential()

    def _head_forward(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        indices: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Similarity head forward pass.
        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Detected boxes to apply similarity learning on.
            indices: indices order for images and labels.
        Returns:
            torch.Tensor: embedding after feature extractor.
        """
        image_list = [
            inputs.images[c].tensor for c in range(len(inputs.images))
        ]
        x = self.roi_pooler([torch.cat(image_list, dim=0)], boxes)
        if indices is not None:
            x = x[indices]

        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_embedder(x)
        x = nn.functional.normalize(x)
        return [x]

    def forward_train(
        self,
        inputs: List[InputSample],
        boxes: List[List[Boxes2D]],
        features: Optional[List[FeatureMaps]],
        targets: List[LabelInstances],
    ) -> Tuple[LossesType, Optional[List[SamplingResult]]]:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched, including
                possible reference views. The keyframe is at index 0.
            boxes: Detected boxes to apply similarity learning on.
            features: Input feature maps. Batched, including possible
                reference views. The keyframe is at index 0.
            targets: Corresponding targets to each InputSample.

        Returns:
            LossesType: A dict containing the cosine_similarity cross
            entropy loss
            None
        """
        box_idx = 0  # We only use the keyframe. Ignore reference views

        # Extract class ids for each box
        class_ids = torch.cat(
            [label.class_ids for label in boxes[box_idx]], dim=0
        ).to(dtype=torch.long)
        # Do not sample more boxes than self.max_boxes_num
        batch_size = min(self.max_boxes_num, len(class_ids))
        indices = torch.randperm(len(class_ids))[:batch_size]
        class_ids = class_ids[indices]

        if (
            batch_size <= 1
        ):  # @todo(zrene) there must be a cleaner solution. Batch size is
            # highly unstable!
            rank_zero_warn(
                "Deepsort Similarity Head: Got Batch size <=1. Skipping this "
                "sample!"
            )
            return {
                "cosine_similarity_ce_loss": torch.tensor(
                    [0.0], requires_grad=True, device=self.classifier.device
                )
            }, None

        x = self._head_forward(inputs[box_idx], boxes[box_idx], indices)
        cls_score = self.classifier(torch.cat(x, dim=0))
        cls_score_softmax = torch.nn.functional.softmax(cls_score, dim=1)
        loss = self.classifier_loss(cls_score_softmax, class_ids)
        return {"cosine_similarity_ce_loss": loss}, None

    def forward_test(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        features: Optional[FeatureMaps],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Input boxes to compute similarity embedding for.
            features: Input feature maps. Batched.

        Returns:
            List[torch.Tensor]: Similarity embeddings (one vector per box, one
            tensor per batch element).
        """
        return self._head_forward(inputs, boxes)
