"""SORT model definition."""
from typing import List

import torch

from vist.model import BaseModel, BaseModelConfig, build_model
from vist.model.track.graph import TrackGraphConfig, build_track_graph
from vist.struct import InputSample, LossesType, ModelOutput


class SORTConfig(BaseModelConfig, extra="allow"):
    """SORT config."""

    detection: BaseModelConfig
    track_graph: TrackGraphConfig


class SORT(BaseModel):
    """SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = SORTConfig(**cfg.dict())
        self.detector = build_model(self.cfg.detection)
        self.track_graph = build_track_graph(self.cfg.track_graph)

    @property
    def device(self) -> torch.device:
        """Get device where input should be moved to."""
        return self.detector.device

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        return self.detector.forward_train(batch_inputs)

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"
        frame_id = batch_inputs[0].metadata.frameIndex
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        output = self.detector.forward_test(
            batch_inputs, postprocess=postprocess
        )
        detections = output["detect"]

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id)

        return dict(detect=detections, track=[tracks])
