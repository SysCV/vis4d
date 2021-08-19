"""SORT model definition."""
from typing import Dict, List

import torch

from vist.model import BaseModel, BaseModelConfig, build_model
from vist.model.deepsort_example.load_predictions import load_predictions
from vist.model.track.graph import TrackGraphConfig, build_track_graph
from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput


class SORTConfig(BaseModelConfig, extra="allow"):
    """SORT config."""

    detection: BaseModelConfig
    track_graph: TrackGraphConfig
    dataset: str
    prediction_path: str


class SORT(BaseModel):
    """SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = SORTConfig(**cfg.dict())
        self.detector = build_model(self.cfg.detection)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.search_dict: Dict[str, Dict[int, Boxes2D]] = dict()

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
        if not self.search_dict:
            self.search_dict = load_predictions(
                self.cfg.dataset, self.cfg.prediction_path
            )

        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"
        frame_id = batch_inputs[0].metadata.frame_index
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = batch_inputs[0].image
        video_name = batch_inputs[0].metadata.video_name
        assert video_name in self.search_dict
        # there might be no detections in one frame, e.g. MOT16-12 frame 443
        if frame_id not in self.search_dict[video_name]:
            detections = [
                Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0)).to(
                    self.device
                )
            ]
        else:
            detections = [
                self.search_dict[video_name][frame_id].to(self.device)
            ]

        ori_wh = (
            batch_inputs[0].metadata.size.width,  # type: ignore
            batch_inputs[0].metadata.size.height,  # type: ignore
        )
        self.postprocess(ori_wh, image.image_sizes[0], detections[0])

        # associate detections, update graph
        if len(detections[0]) == 0:
            tracks = Boxes2D(
                torch.empty(0, 5), torch.empty(0), torch.empty(0)
            ).to(self.device)
        else:
            tracks = self.track_graph(detections[0], frame_id)

        return dict(detect=detections, track=[tracks])  # type:ignore
