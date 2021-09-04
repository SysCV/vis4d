"""deep SORT model definition."""
from typing import Dict, List

import torch
from torchvision.ops import roi_align


from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput
from .base import BaseModel, BaseModelConfig
from .track.graph import TrackGraphConfig, build_track_graph
from .load_predictions import load_predictions
from .deepsort_example.deep import FeatureNet


class DeepSORTConfig(BaseModelConfig, extra="allow"):
    """deep SORT config."""

    detection: BaseModelConfig
    track_graph: TrackGraphConfig
    max_boxes_num: int = 512
    dataset: str
    num_instances: int
    prediction_path: str


class DeepSORT(BaseModel):
    """deep SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__(cfg)
        self.cfg = DeepSORTConfig(**cfg.dict())
        # self.detector = build_detector(self.cfg.detection)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.search_dict: Dict[str, Dict[int, Boxes2D]] = dict()
        self.feature_net = FeatureNet(num_classes=self.cfg.num_instances)

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
        inputs_images = [
            inp[0].image.tensor[0].unsqueeze(0) for inp in batch_inputs
        ]  # no ref views
        inputs_images = torch.cat(inputs_images, dim=0).to(self.device)
        labels = [inp[0].instances for inp in batch_inputs]  # type:ignore

        instance_boxes = torch.empty([0, 5])
        for batch_index, label in enumerate(labels):
            boxes = label.boxes[:, :-1].float()  # type: ignore
            batch_column = torch.ones(len(boxes), 1) * batch_index
            batch_boxes = torch.cat((batch_column, boxes), dim=1)
            instance_boxes = torch.cat((instance_boxes, batch_boxes), dim=0)
        instance_boxes = instance_boxes.to(self.device)
        instance_ids = torch.cat(
            [label.track_ids for label in labels], dim=0  # type:ignore
        ).to(self.device)
        batch_size = min(
            self.cfg.max_boxes_num, len(instance_boxes)  # type:ignore
        )
        indices = torch.randperm(len(instance_boxes))[:batch_size]
        instance_boxes = instance_boxes[indices]
        instance_ids = instance_ids[indices]

        resize = (
            (128, 64)
            if self.cfg.dataset == "MOT16"  # type:ignore
            else (64, 128)
        )
        instance_images = roi_align(
            inputs_images, instance_boxes, resize, aligned=True
        )

        cls_output = self.feature_net(instance_images, train=True)
        instance_ids = instance_ids.long()
        feature_net_loss = torch.nn.functional.cross_entropy(
            cls_output, instance_ids, reduction="mean"
        )
        return feature_net_loss  # type: ignore

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

        frame_id = inputs[0].metadata.frameIndex
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = inputs[0].image
        video_name = inputs[0].metadata.videoName
        assert video_name in self.search_dict
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
            inputs[0].metadata.size.width,  # type: ignore
            inputs[0].metadata.size.height,  # type: ignore
        )
        self.postprocess(ori_wh, image.image_sizes[0], detections[0])

        # associate detections, update graph
        if len(detections[0]) == 0:
            tracks = Boxes2D(
                torch.empty(0, 5), torch.empty(0), torch.empty(0)
            ).to(self.device)
        else:
            instance_boxes = detections[0].boxes[:, :-1]
            image_tensor = image.tensor.to(self.device)
            resize = (
                (128, 64)
                if self.cfg.dataset == "MOT16"  # type:ignore
                else (64, 128)
            )
            instance_images = roi_align(
                image_tensor, [instance_boxes], resize, aligned=True
            )
            det_features = self.feature_net(instance_images, train=False)
            tracks = self.track_graph(detections[0], frame_id, det_features)
        return dict(detect=detections, track=[tracks])  # type:ignore
