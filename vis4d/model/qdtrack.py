"""Quasi-dense instance similarity learning model."""
import pickle
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common.io.hdf5 import HDF5Backend
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
    Boxes2D,
    FeatureMaps,
    InputSample,
    Losses,
    ModelOutput,
)


class QDTrack(nn.Module):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(
        self,
        detection: BaseDetector,
        similarity: BaseSimilarityHead,
        track_graph: BaseTrackGraph,
        inference_result_path: Optional[str] = None,
    ) -> None:
        """Init."""
        super().__init__()
        self.detector = detection
        assert isinstance(
            self.detector, (BaseTwoStageDetector, BaseOneStageDetector)
        )
        self.similarity_head = similarity
        self.track_graph = track_graph
        assert (
            self.detector.category_mapping is not None
        ), "Need category mapping for qdtrack!"
        self.cat_mapping = {
            v: k for k, v in self.detector.category_mapping.items()
        }
        self.with_mask = getattr(self.detector, "with_mask", False)

        self.inference_result_path = inference_result_path
        if self.inference_result_path is not None:
            self.data_backend = HDF5Backend()

    def _run_heads_train(
        self,
        key_images: torch.Tensor,
        ref_images: List[torch.Tensor],
        key_x: FeatureMaps,
        ref_x: List[FeatureMaps],
    ) -> Tuple[Losses, List[Boxes2D], List[List[Boxes2D]]]:
        """Get detection and tracking losses."""
        key_targets, ref_targets = key_images.targets, [
            x.targets for x in ref_inputs
        ]

        key_proposals: Optional[List[Boxes2D]]
        if isinstance(self.detector, BaseTwoStageDetector):
            # proposal generation
            rpn_losses, key_proposals = self.detector.generate_proposals(
                key_images, key_x, key_targets
            )
            with torch.no_grad():
                ref_proposals = [
                    self.detector.generate_proposals(inp, x, tgt)[1]
                    for inp, x, tgt in zip(ref_images, ref_x, ref_targets)
                ]

            # roi head
            assert isinstance(self.detector, BaseTwoStageDetector)
            roi_losses, _ = self.detector.generate_detections(
                key_images,
                key_x,
                key_proposals,
                key_targets,
            )
            det_losses = {**rpn_losses, **roi_losses}
        else:
            # one-stage detector
            det_losses, key_proposals = self.detector.generate_detections(
                key_images, key_x, key_targets
            )
            assert key_proposals is not None
            ref_proposals = []
            with torch.no_grad():
                for inp, x, tgt in zip(ref_images, ref_x, ref_targets):
                    ref_p = self.detector.generate_detections(inp, x, tgt)[1]
                    assert ref_p is not None
                    ref_proposals.append(ref_p)

        # track head
        track_losses, _ = self.similarity_head(
            [key_inputs, *ref_images],
            [key_proposals, *ref_proposals],
            [key_x, *ref_x],
            [key_targets, *ref_targets],
        )
        return {**det_losses, **track_losses}, key_proposals, ref_proposals

    def debug_logging(self, logger) -> Dict[str, torch.Tensor]:
        """Logging for debugging"""
        # from vis4d.vis.track import imshow_bboxes
        # for ref_inp, ref_props in zip(ref_inputs, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_inp.images, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

    def _run_heads_test(
        self, inputs: InputSample, feat: FeatureMaps
    ) -> Tuple[ModelOutput, List[torch.Tensor]]:
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

        outs = {"detect": [d.clone() for d in detections]}  # type: ignore
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
        predictions,
        embeddings: List[torch.Tensor],
    ) -> ModelOutput:
        """Associate detections, update track graph."""
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        outs = {"track": tracks.boxes2d}  # type: ignore
        if self.with_mask:
            outs["seg_track"] = tracks.instance_masks

        postprocess_predictions(
            inputs, outs, self.detector.clip_bboxes_to_image
        )
        return predictions_to_scalabel(outs, self.cat_mapping)

    def forward(
        self, batch_inputs: List[InputSample]
    ) -> Union[Losses, ModelOutput]:
        """Forward function of QDTrack."""
        if self.training:
            return self.forward_train(batch_inputs)
        return self.forward_test(batch_inputs)

    def forward_train(self, batch_inputs: List[InputSample]) -> Losses:
        """Forward function for training."""
        key_inputs, ref_inputs = split_key_ref_inputs(batch_inputs)

        # from vis4d.vis.image import imshow_bboxes
        # for batch_i, key_inp in enumerate(key_inputs):
        #    imshow_bboxes(
        #        key_inp.images.tensor[0], key_inp.targets.boxes2d[0]
        #    )
        #    for ref_i, ref_inp in enumerate(ref_inputs):
        #        imshow_bboxes(
        #            ref_inp[batch_i].images.tensor[0],
        #            ref_inp[batch_i].targets.boxes2d[0],
        #        )

        # TODO temporary connector code
        key_images = key_inputs.images.tensor
        ref_images = [inp.images.tensor for inp in ref_inputs]

        # feature extraction
        key_x = self.detector.extract_features(key_images)
        ref_x = [self.detector.extract_features(im) for im in ref_images]

        losses, _, _ = self._run_heads_train(
            key_images, ref_images, key_x, ref_x
        )
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
            predictions = predictions.to(batch_inputs[0].device)
            embeddings = [e.to(batch_inputs[0].device) for e in embeddings]

        outs.update(self._track(batch_inputs[0], predictions, embeddings))
        return outs

    def forward(
        self, batch_inputs: List[InputSample]
    ) -> Union[LossesType, ModelOutput]:
        """Forward."""
        if self.training:
            return self.forward_train(batch_inputs)
        return self.forward_test(batch_inputs)
