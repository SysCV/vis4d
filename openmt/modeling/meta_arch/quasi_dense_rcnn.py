"""Faster R-CNN for quasi-dense instance similarity learning."""

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from detectron2.modeling import GeneralizedRCNN
from detectron2.structures import Instances


class QDGeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN for quasi-dense instance similarity learning.
    Inherits from GeneralizedRCNN in detectron2, which supports:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    We extend this architecture to be compatible to instance embedding
    learning across multiple frames.
    """

    def __init__(self, track_config, detector_config):
        super().__init__()
        self.d2_detector = GeneralizedRCNN(detector_config)

    def select_keyframe(self, batch_size):
        """Keyframe selection.
        Strategies:
        - Random
        - First frame
        - Last frame
        """
        if self.keyframe_selection == "random":
            key_index = random.randint(0, batch_size)
        elif self.keyframe_selection == "first":
            key_index = 0
        elif self.keyframe_selection == "last":
            key_index = batch_size - 1
        else:
            raise NotImplementedError(
                f"Keyframe selection strategy "
                f"{self.keyframe_selection} not "
                f"implemented"
            )

        ref_indices = list(range(batch_size))
        ref_indices.remove(key_index)

        return key_index, ref_indices

    def forward(
        self,
        batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]],
    ):
        """Forward pass function."""

        if not self.training:
            # During inference, we only feed one frame at a time, therefore
            # only hand over first (and only) element to inference
            return self.inference(batch_inputs[0])

        # preprocess input
        batched_images = [self.preprocess_image(inp) for inp in batch_inputs]

        batch_size = len(batch_inputs[0])
        key_index, ref_indices = self.select_keyframe(batch_size)
        key_inputs = batched_images[key_index]
        ref_inputs = [batched_images[i] for i in ref_indices]

        if "instances" in batch_inputs[0][0]:
            key_targets = [x["instances"].to(self.device) for x in key_inputs]
            ref_targets = [
                [x["instances"].to(self.device) for x in ref_in]
                for ref_in in ref_inputs
            ]
        else:
            key_targets = ref_targets = None

        # backbone
        key_x = self.backbone(key_inputs.tensor)
        ref_x = [self.backbone(ref_im.tensor) for ref_im in ref_inputs]

        # rpn stage
        key_proposals, rpn_losses = self.proposal_generator(key_x)
        ref_proposals = [self.proposal_generator(x)[0] for x in ref_x]

        # detection head(s)
        _, detect_losses = self.roi_heads(
            key_inputs,
            key_x,
            key_proposals,
            key_targets,
        )

        # track head
        targets, key_embeddings = self.track_head(
            key_inputs, key_x, key_proposals, key_targets
        )
        ref_embeddings = []
        for input, x, proposal, target in zip(
            ref_inputs, ref_x, ref_proposals, ref_targets
        ):
            embeddings = self.track_head(input, x, proposal, target)
            ref_embeddings += [embeddings]

        track_losses = self.track_head.matching_loss(
            key_embeddings, ref_embeddings
        )

        losses = dict()
        losses.update(rpn_losses)
        losses.update(detect_losses)
        losses.update(track_losses)
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
