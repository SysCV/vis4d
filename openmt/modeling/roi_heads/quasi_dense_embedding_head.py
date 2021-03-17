"""RoI Heads definition for quasi-dense instance similarity learning"""
from math import prod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from detectron2.layers.batch_norm import get_norm
from detectron2.layers.wrappers import Conv2d
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList

from openmt.config import RoIHead as RoIHeadConfig
from openmt.core.bbox.matchers import build_matcher
from openmt.core.bbox.samplers import build_sampler
from openmt.modeling.roi_heads import BaseRoIHead
from openmt.structures import Boxes2D


class QDRoIHead(BaseRoIHead):
    """Instance embedding head for quasi-dense similarity learning."""

    def __init__(self, cfg: RoIHeadConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.sampler = build_sampler(cfg.proposal_sampler)
        self.matcher = build_matcher(cfg.proposal_matcher)
        self.roi_pooler = ROIPooler(
            output_size=cfg.pooler_resolution,
            scales=[1 / s for s in cfg.pooler_strides],
            sampling_ratio=cfg.pooler_sampling_ratio,
            pooler_type=cfg.pooler_type,
        )

        self.convs, self.fcs, last_layer_dim = self._init_embedding_head(cfg)
        self.fc_embed = nn.Linear(last_layer_dim, cfg.embedding_dim)
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

    def _init_embedding_head(cls, cfg):
        """Init embedding head."""
        last_layer_dim = cfg.in_dim
        # add branch specific conv layers
        convs = nn.ModuleList()
        if cfg.num_convs > 0:
            for i in range(cfg.num_convs):
                conv_in_dim = last_layer_dim if i == 0 else cfg.conv_out_dim
                convs.append(
                    Conv2d(
                        conv_in_dim,
                        cfg.conv_out_dim,
                        kernel_size=3,
                        padding=1,
                        norm=get_norm(cfg.norm, cfg.conv_out_dim),
                        activation=nn.ReLU(),
                    )
                )
            last_layer_dim = cfg.conv_out_dim

        fcs = nn.ModuleList()
        if cfg.num_fcs > 0:
            last_layer_dim *= prod(cfg.pooler_resolution)
            for i in range(cfg.num_fcs):
                fc_in_dim = last_layer_dim if i == 0 else cfg.fc_out_dim
                fcs.append(
                    nn.Sequential(
                        nn.Linear(fc_in_dim, cfg.fc_out_dim), nn.ReLU()
                    )
                )
            last_layer_dim = cfg.fc_out_dim
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
    ) -> Tuple[List[Boxes2D], Optional[List[Boxes2D]]]:
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

        # TODO refactor proposals once converted to our format
        x = self.roi_pooler(
            list(features.values())[:-1], [p.proposal_boxes for p in proposals]
        )

        # convs
        if self.cfg.num_convs > 0:
            for i, conv in enumerate(self.convs):
                x = conv(x)
        x = x.view(x.size(0), -1)

        # fcs
        if self.cfg.num_fcs > 0:
            for i, fc in enumerate(self.fcs):
                x = fc(x)

        return self.fc_embed(x), targets
