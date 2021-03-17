"""Faster R-CNN for quasi-dense instance similarity learning."""

import random
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.modeling import GeneralizedRCNN
from detectron2.structures import Instances

from openmt.config import Config
from openmt.core.track import cosine_similarity
from openmt.detect import to_detectron2

from ..roi_heads import build_roi_head
from .base_arch import BaseMetaArch


class QDGeneralizedRCNN(BaseMetaArch):
    """
    Generalized R-CNN for quasi-dense instance similarity learning.
    Inherits from GeneralizedRCNN in detectron2, which supports:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    We extend this architecture to be compatible to instance embedding
    learning across multiple frames.
    """

    def __init__(self, cfg: Config) -> None:
        """Init."""
        super().__init__()
        detector_config = to_detectron2(cfg)
        self.d2_detector = GeneralizedRCNN(detector_config)
        self.track_head = build_roi_head(cfg.tracking.embedding_head)
        self.keyframe_selection = cfg.tracking.keyframe_selection

    @property
    def device(self):
        """Get device where model input should be moved to."""
        return self.d2_detector.pixel_mean.device

    def select_keyframe(self, sequence_length):
        """Keyframe selection.
        Strategies:
        - Random
        - First frame
        - Last frame
        """
        if self.keyframe_selection == "random":
            key_index = random.randint(0, sequence_length - 1)
        elif self.keyframe_selection == "first":
            key_index = 0
        elif self.keyframe_selection == "last":
            key_index = sequence_length - 1
        else:
            raise NotImplementedError(
                f"Keyframe selection strategy "
                f"{self.keyframe_selection} not "
                f"implemented"
            )

        ref_indices = list(range(sequence_length))
        ref_indices.remove(key_index)

        return key_index, ref_indices

    def forward(self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]):
        """Forward pass function."""

        # TODO change once new dataloader finished
        batch_inputs = [batch_inputs]

        if not self.training:  # TODO change
            return self.inference(batch_inputs[0])

        # preprocess input
        batched_images = [
            self.d2_detector.preprocess_image(inp) for inp in batch_inputs
        ]

        # split into key / ref pairs
        sequence_length = len(batch_inputs)
        key_index, ref_indices = self.select_keyframe(sequence_length)
        ref_indices = [0]  # TODO delete
        key_inputs = batched_images[key_index]
        ref_inputs = [batched_images[i] for i in ref_indices]

        # prepare targets
        if "instances" in batch_inputs[0][0]:
            key_targets = [
                x["instances"].to(self.device) for x in batch_inputs[key_index]
            ]
            ref_targets = [
                [x["instances"].to(self.device) for x in batch_inputs[i]]
                for i in ref_indices
            ]
        else:
            key_targets = ref_targets = None

        # backbone
        key_x = self.d2_detector.backbone(key_inputs.tensor)
        ref_x = [
            self.d2_detector.backbone(ref_im.tensor) for ref_im in ref_inputs
        ]

        # rpn stage
        key_proposals, rpn_losses = self.d2_detector.proposal_generator(
            key_inputs, key_x, key_targets
        )
        ref_proposals = [
            self.d2_detector.proposal_generator(inp, x, target)[0]
            for inp, x, target in zip(ref_inputs, ref_x, ref_targets)
        ]

        # detection head(s)
        _, detect_losses = self.d2_detector.roi_heads(
            key_inputs,
            key_x,
            key_proposals,
            key_targets,
        )

        # TODO conversion between detectron internal format Instances /
        #  Boxes to ours (Boxes2D)

        # track head
        key_embeddings, key_targets = self.track_head(
            key_inputs, key_x, key_proposals, key_targets
        )
        ref_targets, ref_embeddings = [], []
        for input, x, proposal, target in zip(
            ref_inputs, ref_x, ref_proposals, ref_targets
        ):
            embeddings, targets = self.track_head(input, x, proposal, target)
            ref_embeddings += [embeddings]
            ref_targets += [target]

        track_losses = self.tracking_loss(
            key_embeddings, key_targets, ref_embeddings, ref_targets
        )

        losses = dict()
        losses.update(rpn_losses)
        losses.update(detect_losses)
        losses.update(track_losses)
        return losses

    def match(self, key_embeds, ref_embeds):  # TODO rewrite
        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cosine_similarity(
                key_embed,
                ref_embed,
                normalize=False,
                temperature=self.softmax_temp,
            )
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cosine_similarity(key_embed, ref_embed)
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def get_track_targets(self, key_targets, ref_targets):  # TODO rewrite
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_res, ref_res in zip(
            gt_match_indices, key_sampling_results, ref_sampling_results
        ):
            targets = _gt_match_indices.new_zeros(
                (key_res.pos_bboxes.size(0), ref_res.bboxes.size(0)),
                dtype=torch.int,
            )
            _match_indices = _gt_match_indices[key_res.pos_assigned_gt_inds]
            pos2pos = (
                _match_indices.view(-1, 1)
                == ref_res.pos_assigned_gt_inds.view(1, -1)
            ).int()
            targets[:, : pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def tracking_loss(
        self, key_embeddings, key_targets, ref_embeddings, ref_targets
    ):
        """Calculate losses for tracking."""
        losses = dict()

        loss_track = 0.0
        loss_track_aux = 0.0
        dists, cos_dists = self.match(key_embeddings, ref_embeddings)
        track_targets, track_weights = self.get_track_targets(
            key_targets, ref_targets
        )
        for _dists, _cos_dists, _targets, _weights in zip(
            dists, cos_dists, track_targets, track_weights
        ):
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum()
            )
            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses["loss_track"] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses["loss_track_aux"] = loss_track_aux / len(dists)

        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """Inference function."""

        # TODO add inference code

        # init tracker at begin of sequence

        # forward backbone

        # forward RPN --> 1000 proposals out

        # forward RCNN --> detect boxes out

        # forward track head(s) --> instance embedding out

        # associate detections, update tracker

        pass
