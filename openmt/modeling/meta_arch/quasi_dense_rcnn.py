"""Faster R-CNN for quasi-dense instance similarity learning."""

import torch
import random
from typing import Tuple, Dict, Optional, List

from detectron2.structures import Instances
from detectron2.modeling import GeneralizedRCNN


class QDGeneralizedRCNN(GeneralizedRCNN):
    """
        Generalized R-CNN for quasi-dense instance similarity learning.
        Inherits from GeneralizedRCNN in detectron2, which supports:
        1. Per-image feature extraction (aka backbone)
        2. Region proposal generation
        3. Per-region feature extraction and prediction
        We extend this architecture to be compatible to instance embedding
        learning across multiple frames.
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_keyframe(self, batch_size):
        """Keyframe selection.
            Strategies:
            - Random
            - First frame
            - Last frame
        """
        if self.keyframe_selection == 'random':
            key_index = random.randint(0, batch_size)
        elif self.keyframe_selection == 'first':
            key_index = 0
        elif self.keyframe_selection == 'last':
            key_index = batch_size-1
        else:
            raise NotImplementedError(f"Keyframe selection strategy "
                                      f"{self.keyframe_selection} not "
                                      f"implemented")

        ref_indices = list(range(batch_size))
        ref_indices.remove(key_index)

        return key_index, ref_indices


    def forward(self, batched_sequences: Tuple[Tuple[Dict[str,
                                                          torch.Tensor]]],
                ):
        """Forward pass function."""

        if not self.training:
            return self.inference(batched_sequences)

        # preprocess input
        batched_images = [self.preprocess_image(inp) for inp in
                        batched_sequences]

        batch_size = len(batched_sequences[0])
        key_index, ref_indices = self.select_keyframe(batch_size)
        key_inputs = batched_images[key_index]
        ref_inputs = [batched_images[i] for i in ref_indices]

        if "instances" in batched_sequences[0][0]:
            key_targets = [x["instances"].to(self.device) for x in key_inputs]
            ref_targets = [[x["instances"].to(self.device) for x in ref_in]
                           for ref_in in ref_inputs]
        else:
            key_targets = ref_targets = None

        # backbone
        key_x = self.backbone(key_inputs.tensor)
        ref_x = [self.backbone(ref_im.tensor) for ref_im in ref_inputs]

        # rpn stage
        key_proposals, rpn_losses = self.proposal_generator(key_x)
        ref_proposals = [self.proposal_generator(x)[0] for x in ref_x]

        # rcnn stage
        outs = self.roi_heads(key_inputs, ref_inputs, key_x, ref_x,
                              key_proposals, ref_proposals, key_targets,
                              ref_targets)
        _, detect_losses, _, track_losses  = outs  # unpack out

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
        pass

