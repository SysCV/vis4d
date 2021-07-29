"""deep SORT model definition."""
from typing import List, Dict
import torch
from torchvision.ops import roi_align

from examples.deepsort_example.deep import FeatureNet
from examples.deepsort_example.load_predictions import load_predictions
from vist.model import BaseModel, BaseModelConfig
from vist.model.track.graph import TrackGraphConfig, build_track_graph
from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput


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
        super().__init__()
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
        labels = [inp[0].instances for inp in batch_inputs]
        # visualize ground truth on one image
        # from openmt.vis.image import imshow_bboxes

        # imshow_bboxes(
        #     batch_inputs[0][0].image.tensor[0],
        #     batch_inputs[0][0].instances,  # type: ignore
        # )

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
        # print("len(instance_boxes):   ", len(instance_boxes))
        batch_size = min(self.cfg.max_boxes_num, len(instance_boxes))
        # print("len(instance_boxes):   ", len(instance_boxes))
        # print("batch_size:   ", batch_size)
        indices = torch.randperm(len(instance_boxes))[:batch_size]
        instance_boxes = instance_boxes[indices]
        instance_ids = instance_ids[indices]

        resize = (
            (128, 64) if self.cfg.dataset == "MOT16" else (64, 128)
        )  # try (128, 128)
        instance_images = roi_align(
            inputs_images, instance_boxes, resize, aligned=True
        )

        # from PIL import Image
        # import numpy as np

        # for count, instance_img in enumerate(instance_images):
        #     data_im = np.moveaxis(instance_img.cpu().numpy(), 0, -1).astype(
        #         np.uint8
        #     )
        #     img = Image.fromarray(data_im, "RGB")
        #     img.show()
        #     if count == 10:
        #         break
        cls_output = self.feature_net(instance_images, train=True)
        instance_ids = instance_ids.long()
        feature_net_loss = torch.nn.functional.cross_entropy(
            cls_output, instance_ids, reduction="mean"
        )
        return feature_net_loss  # type: ignore

    def forward_test(self, batch_inputs: List[InputSample]) -> ModelOutput:
        """Forward pass during testing stage.

        Returns predictions for each input.
        """
        if not self.search_dict:
            self.search_dict = load_predictions(
                self.cfg.dataset, self.cfg.prediction_path
            )

        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"
        frame_id = batch_inputs[0].metadata.frame_index
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = batch_inputs[0].image
        video_name = batch_inputs[0].metadata.video_name
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

        # visualzie given detections
        # from openmt.vis.image import imshow_bboxes

        # imshow_bboxes(image.tensor[0], detections)

        # # using detectors
        # image, _, _, detections, _ = self.detector(batch_inputs)
        # # use this line only on 6 samples
        # detections[0] = detections[0][detections[0].boxes[:, -1] > 0.5]

        ori_wh = (
            batch_inputs[0].metadata.size.width,  # type: ignore
            batch_inputs[0].metadata.size.height,  # type: ignore
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
            resize = (128, 64) if self.cfg.dataset == "MOT16" else (64, 128)
            instance_images = roi_align(
                image_tensor, [instance_boxes], resize, aligned=True
            )
            det_features = self.feature_net(instance_images, train=False)
            tracks = self.track_graph(detections[0], frame_id, det_features)
        # # visualize tracks
        # from openmt.vis.image import imshow_bboxes

        # imshow_bboxes(image.tensor[0], tracks, frame_id=frame_id)
        # a, _ = torch.sort(detections[0].boxes.cpu().float(), dim=0)
        # b, _ = torch.sort(tracks.boxes.cpu().float(), dim=0)
        # assert torch.allclose(a, b, atol=0.001)
        return dict(detect=detections, track=[tracks])  # type:ignore

        # return dict(detect=detections, track=detections)  # type:ignore
