"""Similarity Head for quasi-dense instance similarity learning."""
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vis4d.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from vis4d.model.losses import BaseLoss, LossConfig, build_loss
from vis4d.struct import Boxes2D, InputSample, LossesType

from .base import BaseSimilarityHead, SimilarityLearningConfig


class DeepSortSimilarityHeadConfig(SimilarityLearningConfig):
    """Deep Sort Similarity Head config."""

    num_instances: int = 625
    num_fcs: int = 1
    proj_dim: int = 32
    fc_out_dim: int = 128
    max_boxes_num: int = 512
    drop_prob: float = 0.6
    loss_cls: Optional[LossConfig]
    roi_align_config: RoIPoolerConfig
    backbone: Optional[str]
    pixel_mean: list = [0.485, 0.456, 0.406]
    pixel_std: list = [0.229, 0.224, 0.225]


class BasicBlock(nn.Module):  # type: ignore
    """Basic build block."""

    def __init__(self, c_in: int, c_out: int, is_downsample: bool = False):
        """Init."""
        super().__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            c_out, c_out, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out),
            )
        elif c_in != c_out:  # pragma: no cover
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.is_downsample = True

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        y = self.conv1(input_x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            input_x = self.downsample(input_x)
        return F.relu(input_x.add(y), True)


def make_layers(
    c_in: int, c_out: int, repeat_times: int = 2, is_downsample: bool = False
) -> nn.Sequential():
    """Make layers."""
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


class DeepSortSimilarityHead(BaseSimilarityHead):
    """DeepSort embedding head for ReID similarity learning."""

    def __init__(self, cfg: SimilarityLearningConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = DeepSortSimilarityHeadConfig(**cfg.dict())
        self.roi_pooler = build_roi_pooler(self.cfg.roi_align_config)
        self.fcs = self._init_layers()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, self.cfg.proj_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cfg.proj_dim),
            nn.ELU(inplace=True),
            nn.Conv2d(
                self.cfg.proj_dim,
                self.cfg.proj_dim,
                3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.cfg.proj_dim),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            make_layers(
                self.cfg.proj_dim, self.cfg.proj_dim, is_downsample=False
            ),
            make_layers(
                self.cfg.proj_dim,
                self.cfg.proj_dim * 2,
                is_downsample=True,
            ),
            make_layers(
                self.cfg.proj_dim * 2,
                self.cfg.proj_dim * 4,
                is_downsample=True,
            ),
        )

        self.classifier = nn.Linear(
            self.cfg.fc_out_dim, self.cfg.num_instances
        )

        self.loss_cls: Optional[BaseLoss] = None
        if self.cfg.loss_cls is not None:
            self.loss_cls = build_loss(self.cfg.loss_cls)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.cfg.pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.cfg.pixel_std).view(-1, 1, 1), False
        )

    def _init_layers(
        self,
    ) -> torch.nn.ModuleList:
        """Init modules of head."""
        fcs = nn.ModuleList()
        if self.cfg.num_fcs > 0:
            input_dim = (
                self.cfg.proj_dim
                * 4
                * np.prod(self.cfg.roi_align_config.resolution)
                // 64
            )
            for _ in range(self.cfg.num_fcs):
                fcs.append(
                    nn.Sequential(
                        nn.Dropout(p=self.cfg.drop_prob),
                        nn.Linear(input_dim, self.cfg.fc_out_dim),
                        nn.BatchNorm1d(self.cfg.fc_out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
        return fcs

    def preprocess_inputs(
        self,
        batch_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """preprocess images samples."""
        batch_inputs = batch_inputs / 255.0
        batch_inputs = (batch_inputs - self.pixel_mean) / self.pixel_std
        return batch_inputs

    def forward(
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Similarity head forward pass.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Detected boxes to apply similarity learning on.
            indices: indices order for images and labels.

        Returns:
            torch.Tensor: embedding after feature backbone extractor.
        """

        x = self.roi_pooler.pool([inputs], boxes)
        if indices is not None:
            x = x[indices]
        x = self.preprocess_inputs(x)

        x = self.backbone(x)

        x = torch.flatten(x, start_dim=1)

        return x

    def forward_train(  # pylint: disable = arguments-renamed
        self,
        inputs: InputSample,
        boxes: List[Boxes2D],
        instance_ids: torch.Tensor,
    ) -> LossesType:
        """Forward pass during training stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            boxes: Detected boxes to apply similarity learning on.
            instance_ids: instance ids GT to calculate similarity loss.

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        batch_size = min(self.cfg.max_boxes_num, len(instance_ids))
        indices = torch.randperm(len(instance_ids))[:batch_size]
        instance_ids = instance_ids[indices]

        x = self.forward(inputs, boxes, indices)

        if self.cfg.num_fcs > 0:
            for fc in self.fcs:
                x = fc(x)

        feats = x
        cls_score = self.classifier(x)

        track_losses = self.loss(
            instance_ids,
            cls_score,
            feats,
        )

        return track_losses

    def forward_test(
        self,
        inputs: InputSample,
        features: Dict[str, torch.Tensor],
        boxes: List[Boxes2D],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: InputSamples (images, metadata, etc). Batched.
            features: Input feature maps. Batched.
            boxes: Input boxes to compute similarity embedding for.

        Returns:
            List[torch.Tensor]: Similarity embeddings (one vector per box, one
            tensor per batch element).
        """
        x = self.forward(inputs, boxes)
        if self.cfg.num_fcs > 0:
            for fc in self.fcs:
                x = fc[1](x)
        return x.div(x.norm(p=2, dim=1, keepdim=True))

    def loss(
        self,
        gt_label: List[torch.Tensor],
        cls_score: List[Boxes2D],
        feats: List[List[torch.Tensor]],  # pylint: disable = unused-argument
    ) -> LossesType:
        """Calculate losses for reid similarity learning.
        use identity loss to learn embedding
        """
        losses = {}

        if self.loss_cls is not None:
            losses["ce_loss"] = self.loss_cls(cls_score, gt_label)

        return losses
