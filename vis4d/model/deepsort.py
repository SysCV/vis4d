"""deep SORT model definition."""
from typing import Dict, List, Optional, Tuple

import torch

from vis4d.model.track.graph.deep_sort_utils import load_predictions
from vis4d.struct import (
    Boxes2D,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    TLabelInstance,
)

from .base import BaseModel, BaseModelConfig, build_model
from .detect import BaseDetectorConfig, BaseTwoStageDetector
from .track.graph import TrackGraphConfig, build_track_graph
from .track.similarity import SimilarityLearningConfig, build_similarity_head
from .utils import predictions_to_scalabel


class DeepSORTConfig(BaseModelConfig):
    """deep SORT config."""

    detection: Optional[BaseDetectorConfig]
    category_mapping: Dict[str, int]
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
            self.detector: BaseTwoStageDetector = build_model(
                self.cfg.detection
            )
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
    ) -> Tuple[List[InputSample], List[List[Boxes2D]]]:
        """Normalize the input images and get instance label."""
        labels = []
        input_image = []
        for inp in batch_inputs:
            inp.images.tensor = (
                inp.images.tensor - self.pixel_mean
            ) / self.pixel_std
            input_image.append(inp.images.tensor)
            assert inp.targets.boxes2d is not None
            labels.append(inp.targets.boxes2d)
        return input_image, labels

    def preprocess_inp(self, inputs: InputSample) -> torch.Tensor:
        """Normalize the input images."""
        inputs.images.tensor = (
            inputs.images.tensor - self.pixel_mean
        ) / self.pixel_std
        return inputs.images.tensor

    def _run_heads_train(
        self,
        inputs: List[InputSample],
    ) -> LossesType:
        """Get detection and tracking losses."""
        input_images, labels = self.preprocess_inputs(inputs)
        track_losses, _ = self.similarity_head(
            input_images, labels, None, labels  # type: ignore
        )
        if self.cfg.detection is None:
            return track_losses  # pragma: no cover
        inputs_detect = inputs[0]
        key_x = self.detector.extract_features(inputs_detect)

        # proposal generation
        rpn_losses, key_proposals = self.detector.generate_proposals(
            inputs_detect, key_x, inputs_detect.targets
        )

        # roi head
        roi_losses, _ = self.detector.generate_detections(
            inputs_detect,
            key_x,
            key_proposals,
            inputs_detect.targets,
        )
        det_losses = {**rpn_losses, **roi_losses}

        return {**det_losses, **track_losses}

    def _run_heads_test(
        self, inputs: InputSample
    ) -> Tuple[ModelOutput, LabelInstances, List[torch.Tensor]]:
        """Get detections and tracks."""
        frame_name = inputs.metadata[0].name
        if self.cfg.detection is None:
            assert (
                self.cfg.prediction_path is not None
            ), "No detector or pre-computed detections defined!"
            self.search_dict = load_predictions(
                self.cfg.prediction_path,
                self.cfg.category_mapping,
            )
            detections = [self.search_dict[frame_name].to(self.device)]

        else:
            feat = self.detector.extract_features(inputs)
            proposals = self.detector.generate_proposals(inputs, feat)
            detections, _ = self.detector.generate_detections(
                inputs, feat, proposals
            )

        outs: Dict[str, List[TLabelInstance]] = {"detect": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long
        outputs = predictions_to_scalabel(
            inputs,
            outs,
            self.cat_mapping,
            self.cfg.detection.clip_bboxes_to_image
            if self.cfg.detection is not None
            else True,
        )

        predictions = LabelInstances(
            detections,
        )
        if self.with_reid:
            input_images = self.preprocess_inp(inputs)
            embeddings = self.similarity_head(input_images, detections, None)
            return outputs, predictions, embeddings

        return outputs, predictions, None  # type: ignore

    def _track(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        embeddings: List[torch.Tensor],
    ) -> ModelOutput:
        """Associate detections, update track graph."""
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        outs: Dict[str, List[TLabelInstance]] = {"track": tracks.boxes2d}  # type: ignore # pylint: disable=line-too-long
        return predictions_to_scalabel(
            inputs,
            outs,
            self.cat_mapping,
            self.cfg.detection.clip_bboxes_to_image
            if self.cfg.detection is not None
            else True,
        )

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        if self.with_reid:
            losses = self._run_heads_train(batch_inputs)
        else:
            raise NotImplementedError
        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"
        outs, predictions, embeddings = self._run_heads_test(batch_inputs[0])
        outs.update(self._track(batch_inputs[0], predictions, embeddings))
        return outs
