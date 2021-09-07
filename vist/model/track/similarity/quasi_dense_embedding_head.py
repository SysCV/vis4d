"""Similarity Head for quasi-dense instance similarity learning."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from vist.common.bbox.matchers import MatcherConfig, build_matcher
from vist.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from vist.common.bbox.samplers import (
    SamplerConfig,
    SamplingResult,
    build_sampler,
)
from vist.common.layers import Conv2d
from vist.model.losses import BaseLoss, LossConfig, build_loss
from vist.struct import Boxes2D, Images, LossesType

from ..utils import cosine_similarity
from .base import BaseSimilarityHead, SimilarityLearningConfig


class QDSimilarityHeadConfig(SimilarityLearningConfig):
    """Quasi-dense Similarity Head config."""

    in_dim: int
    num_convs: int
    conv_out_dim: int
    conv_has_bias: bool
    num_fcs: int
    fc_out_dim: int
    embedding_dim: int
    norm: str
    num_groups: int = 32
    proposal_append_gt: bool
    softmax_temp: float = -1.0
    in_features: List[str] = ["p2", "p3", "p4", "p5"]
    track_loss: LossConfig
    track_loss_aux: Optional[LossConfig]
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

        self.track_loss = build_loss(self.cfg.track_loss)
        self.track_loss_aux: Optional[BaseLoss] = None
        if self.cfg.track_loss_aux is not None:
            self.track_loss_aux = build_loss(self.cfg.track_loss_aux)

        self._init_weights()

    def _init_weights(self) -> None:
        """Init weights of modules in head."""
        for m in self.convs:
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # pragma: no cover

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
        norm = getattr(nn, self.cfg.norm)
        if norm == nn.GroupNorm:
            norm = lambda x: nn.GroupNorm(self.cfg.num_groups, x)
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
                        bias=self.cfg.conv_has_bias,
                        norm=norm(self.cfg.conv_out_dim),
                        activation=nn.ReLU(inplace=True),
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
                        nn.Linear(fc_in_dim, self.cfg.fc_out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            last_layer_dim = self.cfg.fc_out_dim
        return convs, fcs, last_layer_dim

    @torch.no_grad()  # type: ignore
    def match_and_sample_proposals(
        self, proposals: List[Boxes2D], targets: List[Boxes2D]
    ) -> SamplingResult:
        """Match proposals to targets and subsample."""
        if self.cfg.proposal_append_gt:
            proposals = [
                Boxes2D.cat([p, t]) for p, t in zip(proposals, targets)
            ]
        matching = self.matcher.match(proposals, targets)
        return self.sampler.sample(matching, proposals, targets)

    def forward(
        self, features: Dict[str, torch.Tensor], boxes: List[Boxes2D]
    ) -> List[torch.Tensor]:
        """Similarity head forward pass."""
        features_list = [features[f] for f in self.cfg.in_features]
        x = self.roi_pooler.pool(features_list, boxes)
        # convs
        if self.cfg.num_convs > 0:
            for conv in self.convs:
                x = conv(x)

        # fcs
        x = torch.flatten(x, start_dim=1)
        if self.cfg.num_fcs > 0:
            for fc in self.fcs:
                x = fc(x)

        embeddings = self.fc_embed(x).split(
            [len(b) for b in boxes]
        )  # type: List[torch.Tensor]
        return embeddings

    def forward_train(
        self,
        inputs: Union[List[Images], List[Dict[str, torch.Tensor]]],
        boxes: List[List[Boxes2D]],
        targets: List[List[Boxes2D]],
    ) -> Tuple[LossesType, Optional[List[SamplingResult]]]:
        """Forward pass during training stage.

        Args:
            inputs: Either images or feature maps. Batched, including possible
                reference views. The keyframe is assumed to be at index 0.
            boxes: Detected boxes to apply similarity learning on.
            targets: Target boxes with tracking identities.

        Returns:
            LossesType: A dict of scalar loss tensors.
             Optional[List[SamplingResult]]: Sampling result for
        """
        sampling_results, sampled_boxes, sampled_targets = [], [], []
        for i, (box, tgt) in enumerate(zip(boxes, targets)):
            sampling_result = self.match_and_sample_proposals(box, tgt)
            sampling_results.append(sampling_result)

            sampled_box = sampling_result.sampled_boxes
            sampled_tgt = sampling_result.sampled_targets
            if i == 0:  # we assume the keyframe is first
                positives = [l == 1 for l in sampling_result.sampled_labels]
                sampled_box = [b[p] for b, p in zip(sampled_box, positives)]
                sampled_tgt = [t[p] for t, p in zip(sampled_tgt, positives)]

            sampled_boxes.append(sampled_box)
            sampled_targets.append(sampled_tgt)

        embeddings = []
        for feat, box in zip(inputs, sampled_boxes):
            assert isinstance(
                feat, dict
            ), "QDSimilarityHead expects feature maps as input!"
            embeddings.append(self.forward(feat, box))

        track_losses = self.loss(
            embeddings[0],
            sampled_targets[0],
            embeddings[1:],
            sampled_targets[1:],
        )
        return track_losses, sampling_results

    def forward_test(
        self,
        inputs: Union[Images, Dict[str, torch.Tensor]],
        boxes: List[Boxes2D],
    ) -> List[torch.Tensor]:
        """Forward pass during testing stage.

        Args:
            inputs: Model input (batched).
            boxes: Input boxes to compute similarity embedding for.

        Returns:
            List[torch.Tensor]: Similarity embeddings (one vector per box, one
            tensor per batch element).
        """
        assert isinstance(
            inputs, dict
        ), "QDSimilarityHead expects feature maps as input!"
        return self.forward(inputs, boxes)

    @staticmethod
    def get_targets(
        key_targets: List[Boxes2D], ref_targets: List[List[Boxes2D]]
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Create tracking target tensors."""
        # for each reference view
        track_targets, track_weights = [], []
        for ref_target in ref_targets:
            # for each batch element
            curr_targets, curr_weights = [], []
            for key_target, ref_target_ in zip(key_targets, ref_target):
                assert (
                    key_target.track_ids is not None
                    and ref_target_.track_ids is not None
                )
                # target shape: len(key_target) x len(ref_target_)
                target = (
                    key_target.track_ids.view(-1, 1)
                    == ref_target_.track_ids.view(1, -1)
                ).int()
                weight = (target.sum(dim=1) > 0).float()
                curr_targets.append(target)
                curr_weights.append(weight)
            track_targets.append(curr_targets)
            track_weights.append(curr_weights)
        return track_targets, track_weights

    def match(
        self,
        key_embeds: List[torch.Tensor],
        ref_embeds: List[List[torch.Tensor]],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Match key / ref embeddings based on cosine similarity."""
        # for each reference view
        dists, cos_dists = [], []
        for ref_embed in ref_embeds:
            # for each batch element
            dists_curr, cos_dists_curr = [], []
            for key_embed, ref_embed_ in zip(key_embeds, ref_embed):
                dist = cosine_similarity(
                    key_embed,
                    ref_embed_,
                    normalize=False,
                    temperature=self.cfg.softmax_temp,
                )
                dists_curr.append(dist)
                if self.track_loss_aux is not None:
                    cos_dist = cosine_similarity(key_embed, ref_embed_)
                    cos_dists_curr.append(cos_dist)

            dists.append(dists_curr)
            cos_dists.append(cos_dists_curr)
        return dists, cos_dists

    def loss(
        self,
        key_embeddings: List[torch.Tensor],
        key_targets: List[Boxes2D],
        ref_embeddings: List[List[torch.Tensor]],
        ref_targets: List[List[Boxes2D]],
    ) -> LossesType:
        """Calculate losses for tracking.

        Key inputs are of type List[Tensor/Boxes2D] (Lists are length N)
        Ref inputs are of type List[List[Tensor/Boxes2D]] where the lists
        are of length MxN.
        Where M is the number of reference views and N is the
        number of batch elements.
        """
        losses = {}

        loss_track = torch.tensor(0.0, device=key_embeddings[0].device)
        loss_track_aux = torch.tensor(0.0, device=key_embeddings[0].device)
        dists, cos_dists = self.match(key_embeddings, ref_embeddings)
        track_targets, track_weights = self.get_targets(
            key_targets, ref_targets
        )
        # for each reference view
        for curr_dists, curr_cos_dists, curr_targets, curr_weights in zip(
            dists, cos_dists, track_targets, track_weights
        ):
            # for each batch element
            for _dists, _cos_dists, _targets, _weights in zip(
                curr_dists, curr_cos_dists, curr_targets, curr_weights
            ):
                if all(_dists.shape):
                    loss_track += self.track_loss(
                        _dists,
                        _targets,
                        _weights,
                        avg_factor=_weights.sum() + 1e-5,
                    )
                    if self.track_loss_aux is not None:
                        loss_track_aux += self.track_loss_aux(
                            _cos_dists, _targets
                        )

        num_pairs = len(dists) * len(dists[0])
        losses["track_loss"] = loss_track / num_pairs
        if self.track_loss_aux is not None:
            losses["track_loss_aux"] = loss_track_aux / num_pairs

        return losses
