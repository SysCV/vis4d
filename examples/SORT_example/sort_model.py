"""SORT model definition."""
from typing import List

from openmt.model import BaseModel, BaseModelConfig
from openmt.model.detect import BaseDetectorConfig, build_detector
from openmt.model.track.graph import TrackGraphConfig, build_track_graph
from openmt.struct import Boxes2D, InputSample, LossesType


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

    def forward_test(self, batch_inputs: List[InputSample]) -> List[Boxes2D]:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        frame_id = batch_inputs[0].metadata.frame_index
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # detector
        image, _, _, detections, _ = self.detector(batch_inputs)

        # associate detections, update graph
        detections = self.track_graph(detections[0], frame_id)

        ori_wh = tuple(batch_inputs[0].metadata.size)  # type: ignore
        self.postprocess(ori_wh, image.image_sizes[0], detections)  # type: ignore # pylint: disable=line-too-long
        return [detections]
