"""Faster R-CNN for quasi-dense instance similarity learning."""

from typing import Dict, List, Tuple, Union

import torch

from openmt.model.detect import BaseDetectorConfig, build_detector
from openmt.model.track.graph import TrackGraphConfig, build_track_graph
from openmt.model.track.losses import LossConfig, build_loss
from openmt.model.track.similarity import (
    SimilarityLearningConfig,
    build_similarity_head,
)
from openmt.model.track.utils import cosine_similarity, select_keyframe
from openmt.struct import Boxes2D

from .base import BaseModel, BaseModelConfig
from .track.utils import KeyFrameSelection


class QDGeneralizedRCNNConfig(BaseModelConfig):
    """Config for quasi-dense tracking model."""

    keyframe_selection: KeyFrameSelection
    detection: BaseDetectorConfig
    similarity: SimilarityLearningConfig
    track_graph: TrackGraphConfig
    losses: List[LossConfig]
    softmax_temp: float = -1.0


class QDGeneralizedRCNN(BaseModel):
    """Generalized R-CNN for quasi-dense instance similarity learning.

    Inherits from GeneralizedRCNN in detectron2, which supports:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    We extend this architecture to be compatible to instance embedding
    learning across multiple frames.
    """

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__()
        self.cfg = QDGeneralizedRCNNConfig(**cfg.dict())
        self.detector = build_detector(self.cfg.detection)
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.keyframe_selection = self.cfg.keyframe_selection

        self.track_loss = build_loss(self.cfg.losses[0])
        self.track_loss_aux = build_loss(self.cfg.losses[1])

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        return self.detector.device

    def forward_train(
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Forward function for training."""
        # preprocess: group by sequence index instead of batch index, prepare
        batch_inputs = [  # type: ignore
            [batch_inputs[j][i] for j in range(len(batch_inputs))]
            for i in range(len(batch_inputs[0]))
        ]

        batched_images = [
            self.detector.preprocess_image(inp) for inp in batch_inputs
        ]

        # split into key / ref pairs
        sequence_length = len(batch_inputs)
        key_index, ref_indices = select_keyframe(
            sequence_length, self.keyframe_selection
        )
        key_inputs = batched_images[key_index]
        ref_inputs = [batched_images[i] for i in ref_indices]

        # import matplotlib.pyplot as plt
        #
        # def unnormalize(input_img):
        #     color_tensor = input_img.clone()
        #     min, max = (
        #         torch.min(color_tensor, dim=0)[0],
        #         torch.max(color_tensor, dim=0)[0],
        #     )
        #     return color_tensor.sub_(min).div(max - min).mul_(255).int()
        #
        # for i, key_img in enumerate(key_inputs):
        #     print(i)
        #     plt.imshow(unnormalize(key_img.permute(1, 2, 0)))
        #     plt.show()
        #     for ref_img in ref_inputs[i]:
        #         plt.imshow(unnormalize(ref_img.permute(1, 2, 0)))
        #         plt.show()

        # prepare targets
        key_targets = [
            x["instances"].to(self.device) for x in batch_inputs[key_index]
        ]
        ref_targets = [
            [x["instances"].to(self.device) for x in batch_inputs[i]]
            for i in ref_indices
        ]

        key_x, key_proposals, _, det_losses = self.detector(
            key_inputs, key_targets
        )
        ref_out = [
            self.detector(ref_input, ref_target)
            for ref_input, ref_target in zip(ref_inputs, ref_targets)
        ]
        ref_x, ref_proposals = [x[0] for x in ref_out], [x[1] for x in ref_out]

        # track head
        key_embeddings, key_track_targets = self.similarity_head(
            key_inputs,
            key_x,
            key_proposals,
            key_targets,
            filter_negatives=True,
        )
        ref_track_targets, ref_embeddings = [], []
        for inp, x, proposal, target in zip(
            ref_inputs, ref_x, ref_proposals, ref_targets
        ):
            embeds, targets = self.similarity_head(inp, x, proposal, target)
            ref_embeddings += [embeds]
            ref_track_targets += [targets]

        track_losses = self.tracking_loss(
            key_embeddings,
            key_track_targets,
            ref_embeddings,
            ref_track_targets,
        )

        return {**det_losses, **track_losses}

    def match(
        self, key_embeds: torch.Tensor, ref_embeds: List[torch.Tensor]
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

    @staticmethod
    def get_track_targets(
        key_targets: Boxes2D, ref_targets: List[Boxes2D]
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Create tracking target tensors."""
        # for each reference view
        track_targets, track_weights = [], []
        for ref_target in ref_targets:
            # for each batch element
            curr_targets, curr_weights = [], []
            for key_target, ref_target_ in zip(key_targets, ref_target):  # type: ignore # pylint: disable=line-too-long
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

    def tracking_loss(
        self,
        key_embeddings: torch.Tensor,
        key_targets: Boxes2D,
        ref_embeddings: List[torch.Tensor],
        ref_targets: List[Boxes2D],
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """Calculate losses for tracking.

        Each input is of type List[List[Tensor]] where the lists are of
        length MxN where M is the number of reference views
        and N is the number of batch elements.
        """
        losses = dict()

        loss_track = 0.0
        loss_track_aux = 0.0
        dists, cos_dists = self.match(key_embeddings, ref_embeddings)
        track_targets, track_weights = self.get_track_targets(
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
                        _dists, _targets, _weights, avg_factor=_weights.sum()+1e-5
                    )
                    if self.track_loss_aux is not None:
                        loss_track_aux += self.track_loss_aux(_cos_dists, _targets)

        num_pairs = len(dists) * len(dists[0])
        losses["track_loss"] = loss_track / num_pairs
        if self.track_loss_aux is not None:
            losses["track_loss_aux"] = loss_track_aux / num_pairs

        return losses

    def forward_test(  # type: ignore
        self, batch_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> List[Boxes2D]:
        """Forward function during inference."""
        inputs = batch_inputs[0]  # Inference is done using batch size 1

        # init graph at begin of sequence
        if inputs["frame_id"] == 0:
            self.track_graph.reset()

        # detector
        image = self.detector.preprocess_image((inputs,))
        feat, _, detections, _ = self.detector(image)

        # similarity head
        embeddings, _ = self.similarity_head(image, feat, detections, None)

        # associate detections, update graph
        detections = self.track_graph(
            detections[0], inputs["frame_id"], embeddings[0]
        )
        return [detections]
