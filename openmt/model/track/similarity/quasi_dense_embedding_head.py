"""Similarity Head definition for quasi-dense instance similarity learning."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from detectron2.layers.batch_norm import get_norm
from detectron2.layers.wrappers import Conv2d

from openmt.common.bbox.matchers import MatcherConfig, build_matcher
from openmt.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from openmt.common.bbox.samplers import SamplerConfig, build_sampler
from openmt.struct import Boxes2D, Images

from .base import BaseSimilarityHead, SimilarityLearningConfig


class QDSimilarityHeadConfig(SimilarityLearningConfig):
    """Quasi-dense Similarity Head config."""

    num_classes: int
    in_dim: int
    num_convs: int
    conv_out_dim: int
    num_fcs: int
    fc_out_dim: int
    embedding_dim: int
    norm: str
    proposal_append_gt: bool
    in_features: List[str] = ["p2", "p3", "p4", "p5"]
    proposal_pooler: RoIPoolerConfig
    proposal_sampler: SamplerConfig
    proposal_matcher: MatcherConfig


class QDSimilarityHead(BaseSimilarityHead):
    """Instance embedding head for quasi-dense similarity learning."""

    def __init__(self, cfg: SimilarityLearningConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QDSimilarityHeadConfig(**cfg.dict())

        self.sampler = build_sampler(self.cfg.proposal_sampler)
        self.matcher = build_matcher(self.cfg.proposal_matcher)
        self.roi_pooler = build_roi_pooler(self.cfg.proposal_pooler)

        self.convs, self.fcs, last_layer_dim = self._init_embedding_head()
        self.fc_embed = nn.Linear(last_layer_dim, self.cfg.embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Init weights of modules in head."""
        for m in self.convs:
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        for m in self.fcs:
            if isinstance(m[0], nn.Linear):
                nn.init.xavier_uniform_(m[0].weight)
                nn.init.constant_(m[0].bias, 0)

        nn.init.normal_(self.fc_embed.weight, 0, 0.01)
        nn.init.constant_(self.fc_embed.bias, 0)

    def _init_embedding_head(
        self,
    ) -> Tuple[torch.nn.ModuleList, torch.nn.ModuleList, int]:
        """Init modules of head."""
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
            last_layer_dim *= np.prod(self.cfg.proposal_pooler.resolution)
            for i in range(self.cfg.num_fcs):
                fc_in_dim = last_layer_dim if i == 0 else self.cfg.fc_out_dim
                fcs.append(
                    nn.Sequential(
                        nn.Linear(fc_in_dim, self.cfg.fc_out_dim), nn.ReLU()
                    )
                )
            last_layer_dim = self.cfg.fc_out_dim
        return convs, fcs, last_layer_dim

    @torch.no_grad()  # type: ignore
    def match_and_sample_proposals(
        self, proposals: List[Boxes2D], targets: List[Boxes2D]
    ) -> Tuple[List[Boxes2D], List[Boxes2D]]:
        """Match proposals to targets and subsample."""
        if self.cfg.proposal_append_gt:
            proposals = [
                Boxes2D.cat([p, t]) for p, t in zip(proposals, targets)
            ]
        matching = self.matcher.match(proposals, targets)
        return self.sampler.sample(matching, proposals, targets)

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
        filter_negatives: bool = False,
    ) -> Tuple[Tuple[torch.Tensor], Optional[List[Boxes2D]]]:
        """Forward of embedding head.

        We do not return a loss here, since the matching loss needs to
        be computed across all frames. Here, we only run the forward pass
        per frame, as well as proposal sampling and target assignment.
        """
        del images
        features_list = [features[f] for f in self.cfg.in_features]
        if self.training:
            assert targets is not None, "targets required during training"
            proposals, targets = self.match_and_sample_proposals(
                proposals, targets
            )
            if filter_negatives:
                proposals = [
                    p[t.class_ids != -1] for p, t in zip(proposals, targets)  # type: ignore # pylint: disable=line-too-long
                ]
                targets = [t[t.class_ids != -1] for t in targets]  # type: ignore # pylint: disable=line-too-long

        x = self.roi_pooler.pool(features_list, proposals)
        # convs
        if self.cfg.num_convs > 0:
            for conv in self.convs:
                x = conv(x)

        # fcs
        x = torch.flatten(x, start_dim=1)
        if self.cfg.num_fcs > 0:
            for fc in self.fcs:
                x = fc(x)

        embeddings = self.fc_embed(x).split([len(p) for p in proposals])
        return embeddings, targets
