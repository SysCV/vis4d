"""SORT model definition."""
from typing import List, Dict
import json
import torch
from detectron2.data import MetadataCatalog
from openmt.model import BaseModel, BaseModelConfig
from openmt.model.detect import BaseDetectorConfig, build_detector
from openmt.model.track.graph import TrackGraphConfig, build_track_graph
from openmt.struct import Boxes2D, InputSample, LossesType, ModelOutput


class SORTConfig(BaseModelConfig, extra="allow"):
    """SORT config."""

    detection: BaseDetectorConfig
    track_graph: TrackGraphConfig


class SORT(BaseModel):
    """SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = SORTConfig(**cfg.dict())
        self.detector = build_detector(self.cfg.detection)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.search_dict: Dict[str, Dict[int, Boxes2D]] = dict()

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        # SORT only needs to train the detector
        inputs = [inp[0] for inp in batch_inputs]  # no ref views

        # from openmt.vis.image import imshow_bboxes
        # for img, target in zip(images.tensor, targets):
        #     imshow_bboxes(img, target)

        targets = [x.instances.to(self.detector.device) for x in inputs]
        _, _, _, _, det_losses = self.detector(inputs, targets)
        return det_losses  # type: ignore

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        if not self.search_dict:
            from openmt.struct import Boxes2D

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
        # image = batch_inputs[0].image
        # video_name = batch_inputs[0].metadata.video_name
        # frame_index = batch_inputs[0].metadata.frame_index
        # assert video_name in self.search_dict
        # assert frame_index in self.search_dict[video_name]
        # detections = [self.search_dict[video_name][frame_index]]

        # using detectors
        image, _, _, detections, _ = self.detector(batch_inputs)
        ori_wh = (
            batch_inputs[0].metadata.size.width,  # type: ignore
            batch_inputs[0].metadata.size.height,  # type: ignore
        )
        self.postprocess(ori_wh, image.image_sizes[0], detections[0])

        # associate detections, update graph
        detections[0] = detections[0][detections[0].boxes[:, -1] > 0.5]
        tracks = self.track_graph(detections[0], frame_id)

        return dict(detect=detections, track=[tracks])
