"""Deep-Sort model."""
import pickle
from typing import Dict, List, Tuple

import torch

from vis4d.model.base import BaseModel
from vis4d.model.detect import (
    BaseDetector,
    BaseOneStageDetector,
    BaseTwoStageDetector,
)
from vis4d.model.track.graph import BaseTrackGraph
from vis4d.model.track.similarity import BaseSimilarityHead
from vis4d.model.track.utils import split_key_ref_inputs
from vis4d.model.utils import postprocess_predictions, predictions_to_scalabel
from vis4d.struct import (
    ArgsType,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    TLabelInstance,
)


class DeepSort(BaseModel):
    """
    Deep Sort Tracking Model
    """

    def __init__(
        self,
        detection: BaseDetector,
        track_graph: BaseTrackGraph,
        *args: ArgsType,
        similarity: BaseSimilarityHead,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        assert self.category_mapping is not None, "Need category mapping"

        self.detector = detection
        assert isinstance(
            self.detector, (BaseTwoStageDetector, BaseOneStageDetector)
        )
        self.similarity_head = similarity
        self.track_graph = track_graph
        self.cat_mapping = {v: k for k, v in self.category_mapping.items()}

    def _run_heads_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """
        Runs the heads for training
        i.e. extract features for each bounding box and calculate
        classification loss
        """
        target_boxes = [t.targets.boxes2d for t in batch_inputs]

        track_losses, _ = self.similarity_head(
            batch_inputs,
            target_boxes,
            None,  # No features used in deepsort
            [b.targets for b in batch_inputs],
        )

        return {**track_losses}

    def _run_heads_test(
        self, inputs: InputSample, feat: FeatureMaps
    ) -> Tuple[ModelOutput, LabelInstances, List[torch.Tensor]]:
        """Get detections and tracks."""
        if isinstance(self.detector, BaseTwoStageDetector):
            # two-stage detector
            proposals = self.detector.generate_proposals(inputs, feat)
            detections, instance_segms = self.detector.generate_detections(
                inputs, feat, proposals
            )
        else:
            # one-stage detector
            detections = self.detector.generate_detections(inputs, feat)
            instance_segms = None

        # similarity head
        embeddings = self.similarity_head(inputs, detections, feat)

        outs: Dict[str, List[TLabelInstance]] = {  # type: ignore
            "detect": [d.clone() for d in detections]
        }
        if instance_segms is not None:
            outs["ins_seg"] = [s.clone() for s in instance_segms]

        postprocess_predictions(
            inputs,
            outs,
            self.detector.clip_bboxes_to_image,
            self.detector.resolve_overlap,
        )
        outputs = predictions_to_scalabel(outs, self.cat_mapping)

        predictions = LabelInstances(
            detections,
            instance_masks=instance_segms
            if instance_segms is not None
            else None,
        )
        return outputs, predictions, embeddings

    def _track(
        self,
        inputs: InputSample,
        predictions: LabelInstances,
        embeddings: List[torch.Tensor],
    ) -> ModelOutput:
        """Associate detections, update track graph."""
        tracks = self.track_graph(
            inputs, predictions, embeddings=embeddings[0]
        )
        outs: Dict[str, List[TLabelInstance]] = {  # type: ignore
            "track": tracks.boxes2d
        }
        postprocess_predictions(
            inputs, outs, self.detector.clip_bboxes_to_image
        )
        return predictions_to_scalabel(outs, self.cat_mapping)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
        """Forward function for training."""
        key_inputs, _ = split_key_ref_inputs(batch_inputs)
        losses = self._run_heads_train([key_inputs])

        for k, v in self.detector.forward_train([key_inputs]).items():
            losses[k] = v
        return losses

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        result_path = ""
        if self.inference_result_path is not None:
            frame_name = batch_inputs[0].metadata[0].name
            result_path = self.inference_result_path + "/" + frame_name

        if self.inference_result_path is None or not self.data_backend.exists(
            result_path
        ):
            feat = self.detector.extract_features(batch_inputs[0])
            outs, predictions, embeddings = self._run_heads_test(
                batch_inputs[0], feat
            )
            if self.inference_result_path is not None:
                predictions = predictions.to(torch.device("cpu"))
                embeddings = [e.to(torch.device("cpu")) for e in embeddings]
                self.data_backend.set(
                    result_path, pickle.dumps([predictions, embeddings])
                )
        else:
            outs = {}
            predictions, embeddings = pickle.loads(
                self.data_backend.get(result_path)
            )
            predictions = predictions.to(self.device)
            embeddings = [e.to(self.device) for e in embeddings]

        outs.update(self._track(batch_inputs[0], predictions, embeddings))
        return outs
