from typing import Dict, List, Tuple

import torch
from openmt.model import BaseModel, BaseModelConfig
from openmt.struct import Boxes2D
from openmt.model.detect import BaseDetectorConfig, build_detector, d2_utils
from openmt.model.track.graph import TrackGraphConfig, build_track_graph


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
        self, batch_inputs: Tuple[Tuple[Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass during training stage.

        Returns a dict of loss tensors.
        """
        # SORT only needs to train the detector
        batch_inputs = batch_inputs[0]  # no reference views
        images = self.detector.preprocess_image(batch_inputs)  # type: ignore
        targets = [
            d2_utils.target_to_box2d(
                x["instances"].to(self.detector.device)  # type: ignore # pylint: disable=line-too-long
            )
            for x in batch_inputs
        ]

        # from openmt.vis.image import imshow_bboxes
        # for img, target in zip(images.tensor, targets):
        #     imshow_bboxes(img, target)

        _, _, _, det_losses = self.detector(images, targets)
        return det_losses  # type: ignore

    def forward_test(
        self, batch_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> List[Boxes2D]:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        inputs = batch_inputs[0]  # Inference is done using batch size 1

        # init graph at begin of sequence
        if inputs["frame_id"] == 0:
            self.track_graph.reset()

        # detector
        image = self.detector.preprocess_image((inputs,))
        _, _, detections, _ = self.detector(image)

        # associate detections, update graph
        detections = self.track_graph(
            detections[0], inputs["frame_id"]
        )

        self.postprocess(
            (inputs["width"], inputs["height"]),
            (image.tensor.shape[-1], image.tensor.shape[-2]),
            detections,
        )

        return [detections]

