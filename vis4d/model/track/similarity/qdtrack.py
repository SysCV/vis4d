"""Similarity Head for quasi-dense instance similarity learning."""
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from vis4d.common.bbox.matchers import BaseMatcher
from vis4d.common.bbox.poolers import BaseRoIPooler
from vis4d.common.bbox.samplers import (
    BaseSampler,
    SamplingResult,
    match_and_sample_proposals,
)
from vis4d.common.layers import add_conv_branch
from vis4d.common.module import build_module
from vis4d.model.losses import BaseLoss, MultiPosCrossEntropyLoss
from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModuleCfg,
)

from ..utils import cosine_similarity
from .base import BaseSimilarityHead


class QDSimilarityHead(BaseSimilarityHead):
    """Instance embedding head for quasi-dense similarity learning."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        proposal_pooler: Union[ModuleCfg, BaseRoIPooler],
        proposal_sampler: Union[ModuleCfg, BaseSampler],
        proposal_matcher: Union[ModuleCfg, BaseMatcher],
        track_loss: Union[ModuleCfg, BaseLoss] = MultiPosCrossEntropyLoss(),
        track_loss_aux: Optional[Union[ModuleCfg, BaseLoss]] = None,
        in_dim: int = 256,
        num_convs: int = 4,
        conv_out_dim: int = 256,
        conv_has_bias: bool = False,
        num_fcs: int = 1,
        fc_out_dim: int = 1024,
        embedding_dim: int = 256,
        norm: str = "GroupNorm",
        num_groups: int = 32,
        proposal_append_gt: bool = True,
        softmax_temp: float = -1.0,
        in_features: Optional[List[str]] = None,
    ) -> None:
        """Init."""
        super().__init__()
        self.in_dim = in_dim
        self.num_convs = num_convs
        self.conv_out_dim = conv_out_dim
        self.conv_has_bias = conv_has_bias
        self.num_fcs = num_fcs
        self.fc_out_dim = fc_out_dim
        self.norm = norm
        self.num_groups = num_groups
        self.proposal_append_gt = proposal_append_gt
        self.softmax_temp = softmax_temp
        self.in_features = in_features
        if in_features is None:
            self.in_features = ["p2", "p3", "p4", "p5"]

        if isinstance(proposal_sampler, dict):
            self.sampler: BaseSampler = build_module(
                proposal_sampler, bound=BaseSampler
            )
        else:  # pragma: no cover
            self.sampler = proposal_sampler

        if isinstance(proposal_matcher, dict):
            self.matcher: BaseMatcher = build_module(
                proposal_matcher, bound=BaseMatcher
            )
        else:  # pragma: no cover
            self.matcher = proposal_matcher

        if isinstance(proposal_pooler, dict):
            self.roi_pooler: BaseRoIPooler = build_module(
                proposal_pooler, bound=BaseRoIPooler
            )
        else:  # pragma: no cover
            self.roi_pooler = proposal_pooler

        self.convs, self.fcs, last_layer_dim = self._init_embedding_head()
        self.fc_embed = nn.Linear(last_layer_dim, embedding_dim)

        if isinstance(track_loss, dict):
            self.track_loss: BaseLoss = build_module(
                track_loss, bound=BaseLoss
            )
        else:  # pragma: no cover
            self.track_loss = track_loss
        self.track_loss_aux: Optional[BaseLoss] = None
        if track_loss_aux is not None:
            if isinstance(track_loss_aux, dict):
                self.track_loss_aux = build_module(
                    track_loss_aux, bound=BaseLoss
                )
            else:  # pragma: no cover
                self.track_loss_aux = track_loss_aux

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
        convs, last_layer_dim = add_conv_branch(
            self.num_convs,
            self.in_dim,
            self.conv_out_dim,
            self.conv_has_bias,
            self.norm,
            self.num_groups,
        )

        fcs = nn.ModuleList()
        if self.num_fcs > 0:
            last_layer_dim *= np.prod(self.roi_pooler.resolution)
            for i in range(self.num_fcs):
                fc_in_dim = last_layer_dim if i == 0 else self.fc_out_dim
                fcs.append(
                    nn.Sequential(
                        nn.Linear(fc_in_dim, self.fc_out_dim),
                        nn.ReLU(inplace=True),
                    )
                )
            last_layer_dim = self.fc_out_dim
        return convs, fcs, last_layer_dim

    def _head_forward(
        self, features: FeatureMaps, boxes: List[Boxes2D]
    ) -> List[torch.Tensor]:
        """Similarity head forward pass."""
        assert self.in_features is not None
        features_list = [features[f] for f in self.in_features]
        x = self.roi_pooler(features_list, boxes)

        # convs
        if self.num_convs > 0:
            for conv in self.convs:
                x = conv(x)

        # fcs
        x = torch.flatten(x, start_dim=1)
        if self.num_fcs > 0:
            for fc in self.fcs:
                x = fc(x)

        embeddings: List[torch.Tensor] = self.fc_embed(x).split(
            [len(b) for b in boxes]
        )
        return embeddings

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
            LossesType: A dict of scalar loss tensors.
            Optional[List[SamplingResult]]: Sampling results. Key first, then
                reference views.
        """
        assert features is not None, "QDSimilarityHead requires features!"
        sampling_results, sampled_boxes, sampled_targets = [], [], []
        for i, (box, tgt) in enumerate(zip(boxes, targets)):
            sampling_result = match_and_sample_proposals(
                self.matcher,
                self.sampler,
                box,
                tgt.boxes2d,
                self.proposal_append_gt,
            )
            sampling_results.append(sampling_result)

            sampled_box = sampling_result.sampled_boxes
            sampled_tgt = sampling_result.sampled_targets
            positives = [l == 1 for l in sampling_result.sampled_labels]
            if i == 0:  # take only positives for keyframe (assumed at i=0)
                sampled_box = [b[p] for b, p in zip(sampled_box, positives)]
                sampled_tgt = [t[p] for t, p in zip(sampled_tgt, positives)]
            else:  # set track_ids to -1 for all negatives
                for pos, samp_tgt in zip(positives, sampled_tgt):
                    samp_tgt.track_ids[~pos] = -1

            sampled_boxes.append(sampled_box)
            sampled_targets.append(sampled_tgt)

        embeddings = []
        for feat, box in zip(features, sampled_boxes):
            embeddings.append(self._head_forward(feat, box))

        track_losses = self.loss(
            embeddings[0],
            sampled_targets[0],
            embeddings[1:],
            sampled_targets[1:],
        )
        return track_losses, sampling_results

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
        assert features is not None, "QDSimilarityHead requires features!"
        return self._head_forward(features, boxes)

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
                # NOTE: this only works if key only contains positives and all
                # negatives in ref have track_id -1 (see forward_train)
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
                    temperature=self.softmax_temp,
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

        if sum(len(e) for e in key_embeddings) == 0:
            loss_track = sum([e.sum() for e in key_embeddings])
            loss_track_aux = loss_track
        else:
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
            loss_track /= num_pairs
            if self.track_loss_aux is not None:
                loss_track_aux /= num_pairs

        losses["track_loss"] = loss_track
        losses["track_loss_aux"] = loss_track_aux
        return losses
