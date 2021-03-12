"""RoI Heads definition for quasi-dense instance similarity learning"""
import torch
from typing import Dict, List, Optional, Tuple
from detectron2.structures import ImageList, Instances
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads

class QDRoIHeads(StandardROIHeads):
    """Standard RoI Heads + tracking embedding head + matching logic."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

        # TODO add qd-track head / sampling / matching definitions

    def _init_embedding_head(cls, cfg, input_shape):
        """Init embedding head."""
        # TODO track head init
        pass

    def forward(
        self,
            key_inputs: ImageList, ref_inputs: ImageList, key_x: Dict[str,
                                                                      torch.Tensor], ref_x: Dict[str, torch.Tensor],
            key_proposals: List[Instances],
            ref_proposals: List[Instances], key_targets: Optional[List[
                Instances]] = None, ref_targets: Optional[List[
                Instances]] = None) -> Tuple[List[Instances], Dict[str, torch.Tensor], List[Instances], Dict[str, torch.Tensor]]:
        """Forward."""
        detect_result, detect_losses = super().forward(key_inputs, key_x,
                                              key_proposals,
                                  key_targets)

        # TODO track head sampling / assigning / forward / matching / loss
        track_result, track_losses = None, None
        return detect_result, detect_losses, track_result, track_losses