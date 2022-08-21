"""Quasi-dense instance similarity learning model."""
import pickle
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from vis4d.common.io.hdf5 import HDF5Backend
from vis4d.model.detect import FasterRCNN
from vis4d.model.track.graph import QDTrackGraph
from vis4d.model.track.similarity import QDSimilarityHead
from vis4d.model.track.utils import split_key_ref_inputs
from vis4d.model.utils import postprocess_predictions, predictions_to_scalabel
from vis4d.struct import (
    Boxes2D,
    InputSample,
    LabelInstances,
    LossesType,
    ModelOutput,
    NamedTensors,
    TLabelInstance,
)


class QDTrack(nn.Module):
    """QDTrack model - quasi-dense instance similarity learning."""

    def __init__(
        self,
    ) -> None:
        """Init."""
        super().__init__()
        self.similarity_head = QDTrackGraph()
        self.track_graph = QDSimilarityHead()

    def debug_logging(self, logger) -> Dict[str, torch.Tensor]:
        """Logging for debugging"""
        # from vis4d.vis.track import imshow_bboxes
        # for ref_inp, ref_props in zip(ref_inputs, ref_proposals):
        #     for ref_img, ref_prop in zip(ref_inp.images, ref_props):
        #         _, topk_i = torch.topk(ref_prop.boxes[:, -1], 100)
        #         imshow_bboxes(ref_img.tensor[0], ref_prop[topk_i])

    def _run_heads_test(
        self, inputs: InputSample, feat: NamedTensors
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
        tracks = self.track_graph(inputs, predictions, embeddings=embeddings)
        outs: Dict[str, List[TLabelInstance]] = {  # type: ignore
            "track": tracks.boxes2d
        }
        if self.with_mask:
            outs["seg_track"] = tracks.instance_masks

        postprocess_predictions(
            inputs, outs, self.detector.clip_bboxes_to_image
        )
        return predictions_to_scalabel(outs, self.cat_mapping)

    def forward_train(self, batch_inputs: List[InputSample]) -> LossesType:
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
