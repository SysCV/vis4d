"""Faster R-CNN for quasi-dense instance similarity learning."""

import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from detectron2.modeling import GeneralizedRCNN
from detectron2.structures import Instances

from openmt.config import Config
from openmt.core.track import cosine_similarity
from openmt.detect import to_detectron2
from openmt.structures import Boxes2D

from ..losses import build_loss
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
        self.track_head = build_roi_head(cfg.tracking.track_head)
        self.keyframe_selection = cfg.tracking.keyframe_selection
        self.softmax_temp = -1  # TODO move to cfg

        self.track_loss = build_loss(cfg.tracking.losses[0])
        self.track_loss_aux = build_loss(cfg.tracking.losses[1])

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

    def forward(self, batch_inputs: List[List[Dict[str, torch.Tensor]]]):
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

        # detection head(s)  TODO do we need to clone proposals here if e.g. mask head is trained with refined boxes?
        _, detect_losses = self.d2_detector.roi_heads(
            key_inputs,
            key_x,
            key_proposals,
            key_targets,
        )

        key_proposals = proposal_to_box2d(key_proposals)
        key_targets = target_to_box2d(key_targets)
        ref_proposals = [proposal_to_box2d(rp) for rp in ref_proposals]
        ref_targets = [target_to_box2d(rt) for rt in ref_targets]

        # track head
        key_embeddings, key_track_targets = self.track_head(
            key_inputs, key_x, key_proposals, key_targets
        )
        ref_track_targets, ref_embeddings = [], []
        for input, x, proposal, target in zip(
            ref_inputs, ref_x, ref_proposals, ref_targets
        ):
            embeddings, targets = self.track_head(input, x, proposal, target)
            ref_embeddings += [embeddings]
            ref_track_targets += [targets]

        track_losses = self.tracking_loss(
            key_embeddings,
            key_track_targets,
            ref_embeddings,
            ref_track_targets,
        )

        losses = dict()
        losses.update(rpn_losses)
        losses.update(detect_losses)
        losses.update(track_losses)
        return losses

    def match(self, key_embeds, ref_embeds):
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
                if cos_dists is not None:
                    cos_dist = cosine_similarity(key_embed, ref_embed_)
                    cos_dists_curr.append(cos_dist)

            dists.append(dists_curr)
            cos_dists.append(cos_dists_curr)
        return dists, cos_dists

    def get_track_targets(self, key_targets, ref_targets):
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

    def tracking_loss(
        self, key_embeddings, key_targets, ref_embeddings, ref_targets
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
                loss_track += self.track_loss(
                    _dists, _targets, _weights, avg_factor=_weights.sum()
                )
                if self.track_loss_aux is not None:
                    loss_track_aux += self.track_loss_aux(_cos_dists, _targets)

        num_pairs = len(dists) * len(dists[0])
        losses["track_loss"] = loss_track / num_pairs
        if self.track_loss_aux is not None:
            losses["track_loss_aux"] = loss_track_aux / num_pairs

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


"""Detectron2 utils"""  # TODO restructure


def proposal_to_box2d(proposals):
    result = []
    for proposal in proposals:
        boxes, logits = (
            proposal.proposal_boxes.tensor,
            proposal.objectness_logits,
        )
        result.append(
            Boxes2D(
                torch.cat([boxes, logits.unsqueeze(-1)], -1),
                image_wh=proposal.image_size,
            )
        )
    return result


# TODO this does not handle track ids correctly (add them to data first)
def target_to_box2d(targets):
    result = []
    for targets in targets:
        boxes, cls = targets.gt_boxes.tensor, targets.gt_classes
        score = torch.ones((boxes.shape[0], 1), device=boxes.device)
        result.append(
            Boxes2D(
                torch.cat([boxes, score], -1),
                cls,
                torch.arange(0, len(boxes), device=boxes.device),
            )
        )
    return result
