"""Quasi-dense instance similarity learning model."""
import pickle
from typing import Dict, List, Optional, Tuple

import torch

from vis4d.struct import (
    Boxes2D,
    FeatureMaps,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    TLabelInstance,
)

from .base import BaseModel, BaseModelConfig, build_model
from .detect import (
    BaseDetector,
    BaseDetectorConfig,
    BaseOneStageDetector,
    BaseTwoStageDetector,
)
from .track.graph import TrackGraphConfig, build_track_graph
from .track.similarity import SimilarityLearningConfig, build_similarity_head
from .track.utils import split_key_ref_inputs
from .utils import predictions_to_scalabel


class QDTrackConfig(BaseModelConfig):
    """Config for quasi-dense tracking model."""

    detection: BaseDetectorConfig
    similarity: SimilarityLearningConfig
    track_graph: TrackGraphConfig


class QDTrack(BaseModel):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg: QDTrackConfig = QDTrackConfig(**cfg.dict())
        assert self.cfg.category_mapping is not None
        self.cfg.detection.category_mapping = self.cfg.category_mapping
        self.detector: BaseDetector = build_model(self.cfg.detection)
        assert isinstance(
            self.detector, (BaseTwoStageDetector, BaseOneStageDetector)
        )
        self.similarity_head = build_similarity_head(self.cfg.similarity)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.cat_mapping = {v: k for k, v in self.cfg.category_mapping.items()}
        self.with_mask = getattr(self.detector, "with_mask", False)

    def _run_heads_train(
        self,
        key_inputs: InputSample,
        ref_inputs: List[InputSample],
        key_x: FeatureMaps,
        ref_x: List[FeatureMaps],
    ) -> Tuple[LossesType, List[Boxes2D], List[List[Boxes2D]]]:
        """Get detection and tracking losses."""
        key_targets, ref_targets = key_inputs.targets, [
            x.targets for x in ref_inputs
        ]

        key_proposals: Optional[List[Boxes2D]]
        ref_proposals: List[List[Boxes2D]]
        if isinstance(self.detector, BaseTwoStageDetector):
            # two-stage detector
            # proposal generation
            rpn_losses, key_proposals = self.detector.generate_proposals(
                key_inputs, key_x, key_targets
            )
            with torch.no_grad():
                ref_proposals = [
                    self.detector.generate_proposals(inp, x, tgt)[1]
                    for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets)
                ]

            # roi head
            roi_losses, _ = self.detector.generate_detections(
                key_inputs,
                key_x,
                key_proposals,
                key_targets,
            )
            det_losses = {**rpn_losses, **roi_losses}
        else:
            # one-stage detector
            det_losses, key_proposals = self.detector.generate_detections(
                key_inputs, key_x, key_targets
            )
            assert key_proposals is not None
            ref_proposals = []
            with torch.no_grad():
                for inp, x, tgt in zip(ref_inputs, ref_x, ref_targets):
                    ref_p = self.detector.generate_detections(inp, x, tgt)[1]
                    assert ref_p is not None
                    ref_proposals.append(ref_p)

        # from vis4d.vis.track import imshow_bboxes
        # for ref_imgs, ref_props in zip(ref_images, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_imgs, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

        # track head
        track_losses, _ = self.similarity_head(
            [key_inputs, *ref_inputs],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )
        return {**det_losses, **track_losses}, key_proposals, ref_proposals

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

        outs: Dict[str, List[TLabelInstance]] = {"detect": [d.clone() for d in detections]}  # type: ignore # pylint: disable=line-too-long
        if instance_segms is not None:
            outs["ins_seg"] = [s.clone() for s in instance_segms]

        outputs = predictions_to_scalabel(
            inputs,
            outs,
            self.cat_mapping,
            self.cfg.detection.clip_bboxes_to_image,
        )

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
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        outs: Dict[str, List[TLabelInstance]] = {"track": tracks.boxes2d}  # type: ignore # pylint: disable=line-too-long
        if self.with_mask:
            outs["seg_track"] = tracks.instance_masks
        return predictions_to_scalabel(
            inputs,
            outs,
            self.cat_mapping,
            self.cfg.detection.clip_bboxes_to_image,
        )

    def forward_train(
        self,
        batch_inputs: List[InputSample],
    ) -> LossesType:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # from vis4d.vis.image import imshow_bboxes
        # for batch_i, key_inp in enumerate(key_inputs):
        #     imshow_bboxes(key_inp.images.tensor[0], key_inp.boxes2d)
        #     for ref_i, ref_inp in enumerate(ref_inputs):
        #         imshow_bboxes(
        #             ref_inp[batch_i].images.tensor[0],
        #             ref_inp[batch_i].boxes2d,
        #         )

        # feature extraction
        key_x = self.detector.extract_features(key_inputs)
        ref_x = [self.detector.extract_features(inp) for inp in ref_inputs]

        losses, _, _ = self._run_heads_train(
            key_inputs, ref_inputs, key_x, ref_x
        )
        return losses

    def forward_test(
        self,
        batch_inputs: List[InputSample],
    ) -> ModelOutput:
        """Compute model output during inference."""
        assert len(batch_inputs) == 1, "No reference views during test!"
        assert len(batch_inputs[0]) == 1, "Currently only BS=1 supported!"

        result_path = ""
        if self.cfg.inference_result_path is not None:
            frame_name = batch_inputs[0].metadata[0].name
            result_path = self.cfg.inference_result_path + "/" + frame_name

        if (
            self.cfg.inference_result_path is None
            or not self.data_backend.exists(result_path)
        ):
            feat = self.detector.extract_features(batch_inputs[0])
            outs, predictions, embeddings = self._run_heads_test(
                batch_inputs[0], feat
            )
            if self.cfg.inference_result_path is not None:
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
