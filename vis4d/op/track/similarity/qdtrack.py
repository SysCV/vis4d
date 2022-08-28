"""Similarity Head for quasi-dense instance similarity learning."""
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn

from vis4d.common.bbox.poolers import BaseRoIPooler, MultiScaleRoIAlign
from vis4d.common.layers import add_conv_branch
from vis4d.model.losses import EmbeddingDistanceLoss, MultiPosCrossEntropyLoss

from ..utils import cosine_similarity


class QDSimilarityHead(nn.Module):
    """Instance embedding head for quasi-dense similarity learning."""

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
        x = self.roi_pooler(features, boxes)

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


class QDTrackInstanceSimilarityLosses(NamedTuple):  # TODO name
    track_loss: torch.Tensor
    track_loss_aux: torch.Tensor


class QDTrackInstanceSimilarityLoss(nn.Module):
    def __init__(self, softmax_temp: float = -1):
        super().__init__()
        self.softmax_temp = softmax_temp
        self.track_loss = MultiPosCrossEntropyLoss(loss_weight=0.25)
        self.track_loss_aux = EmbeddingDistanceLoss()

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

        Returns:
        """
        losses = {}
        if sum(len(e) for e in key_embeddings) == 0:  # pragma: no cover
            losses["track_loss"] = sum([e.sum() for e in key_embeddings])
            losses["track_loss_aux"] = losses["track_loss"]
            return losses

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
