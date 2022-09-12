"""Quasi-dense embedding similarity based graph."""
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn

from vis4d.common_to_revise.bbox.matchers.max_iou import MaxIoUMatcher
from vis4d.common_to_revise.bbox.poolers import (
    BaseRoIPooler,
    MultiScaleRoIAlign,
)
from vis4d.common_to_revise.bbox.samplers.combined import CombinedSampler
from vis4d.common_to_revise.bbox.utils import bbox_iou
from vis4d.common_to_revise.layers import add_conv_branch
from vis4d.op.loss import EmbeddingDistanceLoss, MultiPosCrossEntropyLoss

from .graph.assignment import TrackIDCounter, greedy_assign
from .graph.matching import calc_bisoftmax_affinity
from .util import cosine_similarity


def get_default_box_sampler() -> CombinedSampler:
    """Get default box sampler of qdtrack."""
    box_sampler = CombinedSampler(
        batch_size=256,
        positive_fraction=0.5,
        pos_strategy="instance_balanced",
        neg_strategy="iou_balanced",
    )
    return box_sampler


def get_default_box_matcher() -> MaxIoUMatcher:
    """Get default box matcher of qdtrack."""
    box_matcher = MaxIoUMatcher(
        thresholds=[0.3, 0.7],
        labels=[0, -1, 1],
        allow_low_quality_matches=False,
    )
    return box_matcher


# @torch.jit.script TODO
class QDTrackAssociation:
    """Data association relying on quasi-dense instance similarity.

    This class assigns detection candidates to a given memory of existing
    tracks and backdrops.
    Backdrops are low-score detections kept in case they have high
    similarity with a high-score detection in succeeding frames.

    Attributes:
        init_score_thr: Confidence threshold for initializing a new track
        obj_score_thr: Confidence treshold s.t. a detection is considered in
        the track / det matching process.
        match_score_thr: Similarity score threshold for matching a detection to
        an existing track.
        memo_backdrop_frames: Number of timesteps to keep backdrops.
        memo_momentum: Momentum of embedding memory for smoothing embeddings.
        nms_backdrop_iou_thr: Maximum IoU of a backdrop with another detection.
        nms_class_iou_thr: Maximum IoU of a high score detection with another
        of a different class.
        with_cats: If to consider category information for tracking (i.e. all
        detections within a track must have consistent category labels).
    """

    def __init__(
        self,
        init_score_thr: float = 0.7,
        obj_score_thr: float = 0.3,
        match_score_thr: float = 0.5,
        nms_backdrop_iou_thr: float = 0.3,
        nms_class_iou_thr: float = 0.7,
        with_cats: bool = True,
    ) -> None:
        """Init."""
        super().__init__()
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats

    def _filter_detections(
        self,
        detections: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Remove overlapping objects across classes via nms.

        Args:
            detections (torch.Tensor): [N, 4] Tensor of boxes.
            scores (torch.Tensor): [N,] Tensor of confidence scores.
            class_ids (torch.Tensor): [N,] Tensor of class ids.
            embeddings (torch.Tensor): [N, C] tensor of appearance embeddings.

        Returns:
            Tuple[torch.Tensor]: filtered detections, scores, class_ids, embeddings, and filtered indices.
        """
        scores, inds = scores.sort(descending=True)
        detections, embeddings, class_ids = (
            detections[inds],
            embeddings[inds],
            class_ids[inds],
        )
        valids = embeddings.new_ones((len(detections),), dtype=torch.bool)
        ious = bbox_iou(detections, detections)
        for i in range(1, len(detections)):
            if scores[i] < self.obj_score_thr:
                thr = self.nms_backdrop_iou_thr
            else:
                thr = self.nms_class_iou_thr

            if (ious[i, :i] > thr).any():
                valids[i] = False
        detections = detections[valids]
        scores = scores[valids]
        class_ids = class_ids[valids]
        embeddings = embeddings[valids]
        return detections, scores, class_ids, embeddings, inds[valids]

    def __call__(
        self,
        detections: torch.Tensor,
        detection_scores: torch.Tensor,
        detection_class_ids: torch.Tensor,
        detection_embeddings: torch.Tensor,
        memory_track_ids: torch.Tensor,
        memory_class_ids: torch.Tensor,
        memory_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process inputs, match detections with existing tracks.

        Args:
            detections (torch.Tensor): [N, 4] detected boxes.
            detection_scores (torch.Tensor): [N,] confidence scores.
            detection_class_ids (torch.Tensor): [N,] class indices.
            detection_embeddings (torch.Tensor): [N, C] appearance embeddings.
            memory_track_ids (torch.Tensor): [M,] track ids in memory.
            memory_class_ids (torch.Tensor): [M,] class indices in memory.
            memory_embeddings (torch.Tensor): [M, C] appearance embeddings in memory.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: track ids of active tracks, selected detection indices corresponding to tracks.
        """
        (
            detections,
            detection_scores,
            detection_class_ids,
            detection_embeddings,
            permute_inds,
        ) = self._filter_detections(
            detections,
            detection_scores,
            detection_class_ids,
            detection_embeddings,
        )
        if len(detections) == 0:
            return torch.empty(
                (0,), dtype=torch.long, device=detections.device
            ), torch.empty((0,), dtype=torch.long, device=detections.device)

        # match if buffer is not empty
        if len(memory_track_ids) > 0:
            affinity_scores = calc_bisoftmax_affinity(
                detection_embeddings,
                memory_embeddings,
                detection_class_ids,
                memory_class_ids,
                self.with_cats,
            )
            ids = greedy_assign(
                detection_scores,
                memory_track_ids,
                affinity_scores,
                self.match_score_thr,
                self.obj_score_thr,
            )
        else:
            ids = torch.full(
                (len(detections),),
                -1,
                dtype=torch.long,
                device=detections.device,
            )

        new_inds = (ids == -1) & (detection_scores > self.init_score_thr)
        ids[new_inds] = TrackIDCounter.get_ids(
            new_inds.sum(), device=ids.device
        )
        return ids, permute_inds


class QDSimilarityHead(nn.Module):
    """Instance embedding head for quasi-dense similarity learning.

    Given a set of input feature maps and RoIs, pool RoI representations from
    feature maps and process them to a per-RoI embeddings vector.
    """

    def __init__(
        self,
        proposal_pooler: Optional[BaseRoIPooler] = None,
        in_dim: int = 256,
        num_convs: int = 4,
        conv_out_dim: int = 256,
        conv_has_bias: bool = False,
        num_fcs: int = 1,
        fc_out_dim: int = 1024,
        embedding_dim: int = 256,
        norm: str = "GroupNorm",
        num_groups: int = 32,
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

        if proposal_pooler is not None:
            self.roi_pooler = proposal_pooler
        else:
            self.roi_pooler = MultiScaleRoIAlign(
                resolution=[7, 7], strides=[4, 8, 16, 32], sampling_ratio=0
            )

        self.convs, self.fcs, last_layer_dim = self._init_embedding_head()
        self.fc_embed = nn.Linear(last_layer_dim, embedding_dim)
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

    def forward(
        self, features: List[torch.Tensor], boxes: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Similarity head forward pass."""
        # take features of strides 4, 8, 16, 32
        x = self.roi_pooler(features[2:6], boxes)

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


class QDTrackInstanceSimilarityLosses(NamedTuple):
    """QDTrack losses return type. Consists of two scalar loss tensors."""

    track_loss: torch.Tensor
    track_loss_aux: torch.Tensor


class QDTrackInstanceSimilarityLoss(nn.Module):
    """Instance similarity loss as in QDTrack.

    Given a number of key frame embeddings and a number of reference frame
    embeddings along with their track identities, compute two losses:
    1. Multi-positive cross-entropy loss.
    2. Cosine similarity loss (auxiliary).
    """

    def __init__(self, softmax_temp: float = -1):
        """Init.

        Args:
            softmax_temp (float, optional): Temperature parameter for multi-positive cross-entropy loss. Defaults to -1.
        """
        super().__init__()
        self.softmax_temp = softmax_temp
        self.track_loss = MultiPosCrossEntropyLoss()
        self.track_loss_aux = EmbeddingDistanceLoss()
        self.track_loss_weight = 0.25

    def forward(
        self,
        key_embeddings: List[torch.Tensor],
        ref_embeddings: List[List[torch.Tensor]],
        key_track_ids: List[torch.Tensor],
        ref_track_ids: List[List[torch.Tensor]],
    ) -> QDTrackInstanceSimilarityLosses:
        """QDTrack instance similarity loss.

        Key inputs are of type List[Tensor/Boxes2D] (Lists are length N)
        Ref inputs are of type List[List[Tensor/Boxes2D]] where the lists
        are of length MxN.
        Where M is the number of reference views and N is the
        number of batch elements.

        NOTE: this only works if key only contains positives and all
        negatives in ref have track_id -1

        Args:
            key_embeddings (List[torch.Tensor]): key frame embeddings.
            ref_embeddings (List[List[torch.Tensor]]): reference frame embeddings.
            key_track_ids (List[torch.Tensor]): associated track ids per embedding in key frame.
            ref_track_ids (List[List[torch.Tensor]]):  associated track ids per embedding in reference frame(s).

        Returns:
            QDTrackInstanceSimilarityLosses: Scalar loss tensors.
        """
        if sum(len(e) for e in key_embeddings) == 0:  # pragma: no cover
            dummy_loss = torch.sum([e.sum() for e in key_embeddings])
            return QDTrackInstanceSimilarityLosses(dummy_loss, dummy_loss)

        loss_track = torch.tensor(0.0, device=key_embeddings[0].device)
        loss_track_aux = torch.tensor(0.0, device=key_embeddings[0].device)
        dists, cos_dists = self._match(key_embeddings, ref_embeddings)
        track_targets, track_weights = self._get_targets(
            key_track_ids, ref_track_ids
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
                    loss_track += (
                        self.track_loss(
                            _dists,
                            _targets,
                            _weights,
                            avg_factor=_weights.sum() + 1e-5,
                        )
                        * self.track_loss_weight
                    )
                    if self.track_loss_aux is not None:
                        loss_track_aux += self.track_loss_aux(
                            _cos_dists, _targets
                        )

        num_pairs = len(dists) * len(dists[0])
        loss_track = loss_track / num_pairs
        loss_track_aux = loss_track_aux / num_pairs

        return QDTrackInstanceSimilarityLosses(
            track_loss=loss_track, track_loss_aux=loss_track_aux
        )

    @staticmethod
    def _get_targets(
        key_track_ids: List[torch.Tensor],
        ref_track_ids: List[List[torch.Tensor]],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Create tracking target tensors."""
        # for each reference view
        track_targets, track_weights = [], []
        for ref_target in ref_track_ids:
            # for each batch element
            curr_targets, curr_weights = [], []
            for key_target, ref_target_ in zip(key_track_ids, ref_target):
                # target shape: len(key_target) x len(ref_target_)
                # NOTE: this only works if key only contains positives and all
                # negatives in ref have track_id -1
                target = (
                    key_target.view(-1, 1) == ref_target_.view(1, -1)
                ).int()
                weight = (target.sum(dim=1) > 0).float()
                curr_targets.append(target)
                curr_weights.append(weight)
            track_targets.append(curr_targets)
            track_weights.append(curr_weights)
        return track_targets, track_weights

    def _match(
        self,
        key_embeds: List[torch.Tensor],
        ref_embeds: List[List[torch.Tensor]],
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Calculate distances for all pairs of key / ref embeddings."""
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
