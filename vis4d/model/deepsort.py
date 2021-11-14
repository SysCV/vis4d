"""deep SORT model definition."""
from typing import Dict, List, Optional

import torch

from vis4d.model.track.deep_sort_utils import load_predictions
from vis4d.struct import Boxes2D, InputSample, LossesType, ModelOutput

from .base import BaseModel, BaseModelConfig
from .track.graph import TrackGraphConfig, build_track_graph
from .track.similarity import SimilarityLearningConfig, build_similarity_head


class DeepSORTConfig(BaseModelConfig):
    """deep SORT config."""

    detection: BaseModelConfig
    track_graph: TrackGraphConfig
    dataset: str
    prediction_path: str
    similarity: Optional[SimilarityLearningConfig]


class DeepSORT(BaseModel):
    """deep SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__(cfg)
        self.cfg = DeepSORTConfig(**cfg.dict())
        assert self.cfg.category_mapping is not None
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        if self.cfg.similarity is not None:
            self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.search_dict: Dict[str, Dict[int, Boxes2D]] = {}
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        return (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        train the feature extractor net
        Returns a dict of loss tensors.
        """
        # inputs = self.preprocess_inputs(batch_inputs)
        inputs_images = [
            inp[0].images.tensor[0].unsqueeze(0) for inp in batch_inputs
        ]  # no ref views
        inputs_images = torch.cat(inputs_images, dim=0).to(self.device)

        labels = []
        for inp in batch_inputs:
            assert inp[0].boxes2d is not None
            labels.append(inp[0].boxes2d[0].to(self.device))
        instance_ids = torch.cat([label.track_ids for label in labels], dim=0)

        track_losses = self.similarity_head.forward_train(
            inputs_images, labels, instance_ids
        )
        return track_losses

    def forward_test(
        self, batch_inputs: List[List[InputSample]]
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        assert len(batch_inputs) == 1, "No reference views during test!"
        inputs = [inp[0] for inp in batch_inputs]
        assert len(inputs) == 1, "Currently only BS=1 supported!"
        if not self.search_dict:
            self.search_dict = load_predictions(
                self.cfg.dataset, self.cfg.prediction_path  # type:ignore
            )

        frame_id = inputs[0].metadata[0].frameIndex
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = inputs[0].images
        video_name = inputs[0].metadata[0].videoName
        # assert video_name in self.search_dict
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

        # # using detectors
        # image, _, _, detections, _ = self.detector(batch_inputs)
        # # use this line only on 6 samples
        # detections[0] = detections[0][detections[0].boxes[:, -1] > 0.5]

        ori_wh = (
            inputs[0].metadata[0].size.width,  # type: ignore
            inputs[0].metadata[0].size.height,  # type: ignore
        )
        # self.postprocess(ori_wh, image.image_sizes[0], detections[0])
        detections[0].postprocess(ori_wh, image.image_sizes[0])
        # associate detections, update graph
        if len(detections[0]) == 0:
            tracks = Boxes2D(
                torch.empty(0, 5), torch.empty(0), torch.empty(0)
            ).to(self.device)
        else:
            if self.cfg.similarity is not None:
                image_tensor = image.tensor.to(self.device)
                det_features = self.similarity_head.forward_test(
                    image_tensor, image_tensor, [detections[0]]
                )
                tracks = self.track_graph(
                    detections[0], frame_id, det_features
                )
            else:
                tracks = self.track_graph(detections[0], frame_id)
        detects = (
            detections[0].to(torch.device("cpu")).to_scalabel(self.cat_mapping)
        )
        tracks_ = tracks.to(torch.device("cpu")).to_scalabel(self.cat_mapping)

        return dict(detect=[detects], track=[tracks_])
