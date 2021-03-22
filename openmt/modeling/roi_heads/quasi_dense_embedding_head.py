"""RoI Heads definition for quasi-dense instance similarity learning"""
from math import prod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from detectron2.layers.batch_norm import get_norm
from detectron2.layers.wrappers import Conv2d
from detectron2.structures import ImageList

from openmt.core.bbox.matchers import MatcherConfig, build_matcher
from openmt.core.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from openmt.core.bbox.samplers import SamplerConfig, build_sampler
from openmt.structures import Boxes2D

from .base_roi_head import BaseRoIHead, RoIHeadConfig


class QDRoIHeadConfig(RoIHeadConfig):
    num_classes: int  # TODO necessary?
    in_dim: int
    num_convs: int
    conv_out_dim: int
    num_fcs: int
    fc_out_dim: int
    embedding_dim: int
    norm: str
    proposal_pooler: RoIPoolerConfig
    proposal_sampler: SamplerConfig
    proposal_matcher: MatcherConfig


class QDRoIHead(BaseRoIHead):
    """Instance embedding head for quasi-dense similarity learning."""

    def __init__(self, cfg: RoIHeadConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QDRoIHeadConfig(**cfg.__dict__)

        self.sampler = build_sampler(self.cfg.proposal_sampler)
        self.matcher = build_matcher(self.cfg.proposal_matcher)
        self.roi_pooler = build_roi_pooler(self.cfg.proposal_pooler)

        self.convs, self.fcs, last_layer_dim = self._init_embedding_head()
        self.fc_embed = nn.Linear(last_layer_dim, self.cfg.embedding_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.convs:
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        for m in self.fcs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def _init_embedding_head(self):
        """Init embedding head."""
        last_layer_dim = self.cfg.in_dim
        # add branch specific conv layers
        convs = nn.ModuleList()
        if self.cfg.num_convs > 0:
            for i in range(self.cfg.num_convs):
                conv_in_dim = (
                    last_layer_dim if i == 0 else self.cfg.conv_out_dim
                )
                convs.append(
                    Conv2d(
                        conv_in_dim,
                        self.cfg.conv_out_dim,
                        kernel_size=3,
                        padding=1,
                        norm=get_norm(self.cfg.norm, self.cfg.conv_out_dim),
                        activation=nn.ReLU(),
                    )
                )
            last_layer_dim = self.cfg.conv_out_dim

        fcs = nn.ModuleList()
        if self.cfg.num_fcs > 0:
            last_layer_dim *= prod(self.cfg.proposal_pooler.resolution)
            for i in range(self.cfg.num_fcs):
                fc_in_dim = last_layer_dim if i == 0 else self.cfg.fc_out_dim
                fcs.append(
                    nn.Sequential(
                        nn.Linear(fc_in_dim, self.cfg.fc_out_dim), nn.ReLU()
                    )
                )
            last_layer_dim = self.cfg.fc_out_dim
        return convs, fcs, last_layer_dim

    def match_and_sample_proposals(self, proposals, targets):
        matching = self.matcher.match(proposals, targets)
        return self.sampler.sample(matching, proposals, targets)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[Boxes2D]]]:
        """Forward of embedding head.
        We do not return a loss here, since the matching loss needs to
        be computed across all frames. Here, we only run the forward pass
        per frame, as well as proposal sampling and target assignment.
        """
        del images

        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals, targets = self.match_and_sample_proposals(
                proposals, targets
            )

        x = self.roi_pooler.pool(list(features.values())[:-1], proposals)

        # convs
        if self.cfg.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)

        # fcs
        x = torch.flatten(x, start_dim=1)
        if self.cfg.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = fc(x)

        embeddings = self.fc_embed(x).split([len(p) for p in proposals])
        return embeddings, targets
