"""deep SORT model definition."""
from typing import Dict, List, Optional

import torch

from vis4d.model.track.graph.deep_sort_utils import load_predictions
from vis4d.struct import Boxes2D, InputSample, LabelInstances, LossesType, \
    ModelOutput

from .base import BaseModel, BaseModelConfig, build_model
from .track.graph import TrackGraphConfig, build_track_graph
from .track.similarity import SimilarityLearningConfig, build_similarity_head


class DeepSORTConfig(BaseModelConfig):
    """deep SORT config."""

    detection: Optional[BaseModelConfig]
    track_graph: TrackGraphConfig
    prediction_path: Optional[str]
    similarity: Optional[SimilarityLearningConfig]
    pixel_mean: List[float] = [123.675, 116.28, 103.53]
    pixel_std: List[float] = [58.395, 57.12, 57.375]


class DeepSORT(BaseModel):
    """deep SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__(cfg)
        self.cfg: DeepSORTConfig = DeepSORTConfig(**cfg.dict())
        assert self.cfg.category_mapping is not None

        if self.cfg.detection is not None:
            self.detector = build_model(self.cfg.detection)
            self.cfg.detection.category_mapping = self.cfg.category_mapping
        if self.cfg.similarity is not None:
            self.similarity_head = build_similarity_head(self.cfg.similarity)
            self.with_reid = True
        else:
            self.with_reid = False
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.search_dict: Dict[str, Boxes2D] = {}
        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.cfg.pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self.cfg.pixel_std).view(-1, 1, 1), False
        )

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        return (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def preprocess_inputs(
        self, batch_inputs: List[InputSample]
    ) -> List[InputSample]:
        """Normalize the input images."""
        inputs_batch = []
        for inp in batch_inputs:
            inp.images.tensor = (inp.images.tensor - self.pixel_mean
                                   ) / self.pixel_std
            inputs_batch.append(inp)
        return inputs_batch

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        train the feature extractor net
        Returns a dict of loss tensors.
        """
        raw_inputs = [inp[0] for inp in batch_inputs]

        inputs = self.preprocess_inputs(raw_inputs, self.with_reid)
        labels = []
        for inp in batch_inputs:
            assert inp[0].boxes2d is not None
            labels.append(inp[0].boxes2d[0].to(self.device))

        track_losses, _ = self.similarity_head.forward_train(
            [inputs.images.tensor], [labels]
        )
        if self.cfg.detection is None:
            return track_losses

        key_x = self.detector.extract_features(inputs)

        # proposal generation
        key_proposals, rpn_losses = self.detector.generate_proposals(
            inputs, key_x
        )

        # roi head
        _, roi_losses, _ = self.detector.generate_detections(
            inputs,
            key_x,
            key_proposals,
            compute_detections=False,
        )

        det_losses = {**rpn_losses, **roi_losses}
        return {**det_losses, **track_losses}

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"
        if self.with_reid:
            inputs = self.detector.preprocess_inputs(batch_inputs[0])
        else:
            inputs = batch_inputs[0]
        frame_id = inputs.metadata[0].frameIndex
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = inputs.images
        frame_name = inputs.metadata[0].name

        if self.cfg.detection is None:
            assert (
                self.cfg.prediction_path is not None
            ), "No detector or pre-computed detections defined!"
            self.search_dict = load_predictions(self.cfg.prediction_path,
                                self.cfg.category_mapping # type: ignore
            )
            detections = [self.search_dict[frame_name].to(self.device)]
        else:
            feat = self.detector.extract_features(inputs)
            proposals, _ = self.detector.generate_proposals(inputs, feat)
            detections, _, _ = self.detector.generate_detections(
                inputs, feat, proposals
            )

        input_size = (
            inputs.metadata[0].size.width,  # type: ignore
            inputs.metadata[0].size.height,  # type: ignore
        )
        detections[0].postprocess(input_size, image.image_sizes[0])
        # associate detections, update graph

        predictions = LabelInstances(
            detections,
        )

        if len(detections[0]) == 0:
            tracks = Boxes2D(  # pragma: no cover
                torch.empty(0, 5), torch.empty(0), torch.empty(0)
            ).to(self.device)
        else:
            if self.with_reid:
                image_tensor = image.tensor.to(self.device)
                det_features = self.similarity_head.forward_test(
                    image_tensor, [detections[0]]
                )
                embeddings = self.similarity_head(inputs, detections, feat)
                tracks = self.track_graph(inputs, predictions,
                                          embeddings=embeddings)
            else:
                tracks = self.track_graph(inputs, predictions)
        detects = (
            detections[0].to(torch.device("cpu")).to_scalabel(self.cat_mapping)
        )
        tracks_ = (
            tracks
            .to(torch.device("cpu"))
            .to_scalabel(self.cat_mapping)
        )
        outputs = dict(detect=[detects])
        outputs["track"] = [tracks_]
        return outputs
