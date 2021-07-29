"""SORT model definition."""
from typing import List, Dict
import json
import torch
from detectron2.data import MetadataCatalog

import torch

from vist.model import BaseModel, BaseModelConfig, build_model
from vist.model.track.graph import TrackGraphConfig, build_track_graph
from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput


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
            self.search_dict = dict()
            given_predictions = json.load(
                open(
                    "weight/predictions.json",
                    "r",
                )
            )

            for prediction in given_predictions:
                video_name = prediction["videoName"]
                frame_index = prediction["frameIndex"]
                if video_name not in self.search_dict:
                    self.search_dict[video_name] = dict()
                boxes = torch.empty((0, 5))
                class_ids = torch.empty((0))
                if "labels" not in prediction:
                    self.search_dict[video_name][frame_index] = Boxes2D(
                        torch.empty((0, 5))
                    )
                else:
                    for label in prediction["labels"]:
                        boxes = torch.cat(
                            (
                                boxes,
                                torch.tensor(
                                    [
                                        label["box2d"]["x1"],
                                        label["box2d"]["y1"],
                                        label["box2d"]["x2"],
                                        label["box2d"]["y2"],
                                        label["score"],
                                    ],
                                ).unsqueeze(0),
                            ),
                            dim=0,
                        )
                        idx_to_class_mapping = MetadataCatalog.get(
                            "bdd100k_sample_val"
                        ).idx_to_class_mapping
                        class_to_idx_mapping = {
                            v: k for k, v in idx_to_class_mapping.items()
                        }
                        class_ids = torch.cat(
                            (
                                class_ids,
                                torch.tensor(
                                    [class_to_idx_mapping[label["category"]]]
                                ),
                            )
                        )

                    self.search_dict[video_name][frame_index] = Boxes2D(
                        boxes, class_ids
                    )

        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"
        frame_id = batch_inputs[0].metadata.frame_index
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = batch_inputs[0].image
        video_name = batch_inputs[0].metadata.video_name
        frame_index = batch_inputs[0].metadata.frame_index
        assert video_name in self.search_dict
        assert frame_index in self.search_dict[video_name]
        detections = [self.search_dict[video_name][frame_index]]

        # # using detectors
        # image, _, _, detections, _ = self.detector(batch_inputs)
        # use this line only on 6 samples
        # detections[0] = detections[0][detections[0].boxes[:, -1] > 0.5]

        ori_wh = (
            batch_inputs[0].metadata.size.width,  # type: ignore
            batch_inputs[0].metadata.size.height,  # type: ignore
        )
        detections = output["detect"]

        # associate detections, update graph
        tracks = self.track_graph(detections[0], frame_id)
        return dict(detect=detections, track=[tracks])  # type:ignore
