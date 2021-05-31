"""deep SORT model definition."""
from typing import List, Dict, Union
import json
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from torchvision.ops import roi_align
from detectron2.data import MetadataCatalog

from deep import FeatureNet
from openmt.model import BaseModel, BaseModelConfig
from openmt.model.detect import BaseDetectorConfig, build_detector
from openmt.model.track.graph import TrackGraphConfig, build_track_graph
from openmt.struct import Boxes2D, InputSample, LossesType, ModelOutput


class DeepSORTConfig(BaseModelConfig, extra="allow"):
    """deep SORT config."""

    detection: BaseDetectorConfig
    track_graph: TrackGraphConfig
    num_boxes_of_training: int = 32
    num_instances: int = 108524
    # featurenet_weight_path: Union[str, None] = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.t7"
    featurenet_weight_path: Union[str, None] = None


class DeepSORT(BaseModel):
    """deep SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = DeepSORTConfig(**cfg.dict())
        self.detector = build_detector(self.cfg.detection)
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.search_dict: Dict[str, Dict[int, Boxes2D]] = dict()
        self.feature_net = FeatureNet(num_classes=self.cfg.num_instances)
        self.feature_extractor = FeatureExtractor(
            self.feature_net, model_weight_path=self.cfg.featurenet_weight_path
        )
        self.norm = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
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
        inputs_images = torch.cat(inputs_images, dim=0)

        labels = [
            inp[0].instances.to(self.detector.device) for inp in batch_inputs
        ]

        instance_boxes = [label.boxes[:, :-1].float() for label in labels]  # type: ignore
        instance_ids = torch.cat([label.track_ids for label in labels], dim=0)  # type: ignore
        instance_images = roi_align(
            inputs_images, instance_boxes, (64, 128), aligned=True
        )
        instance_images = self.norm(instance_images)
        # from PIL import Image
        # import numpy as np
        # for instance_img in instance_images:
        #     data_im = np.moveaxis(instance_img.numpy(), 0, -1).astype(np.uint8)
        #     img = Image.fromarray(data_im, "RGB")
        #     img.show()
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
            self.search_dict = dict()
            given_predictions = json.load(
                open(
                    "weight/predictions.json",
                    "r",
                )
            )

            for prediction in given_predictions:
                video_name = prediction["videoName"]
                frame_index = prediction["frameIndex"]
                if video_name not in self.search_dict:
                    self.search_dict[video_name] = dict()
                boxes = torch.empty((0, 5))
                class_ids = torch.empty((0))
                if "labels" not in prediction:
                    self.search_dict[video_name][frame_index] = Boxes2D(
                        torch.empty((0, 5))
                    )
                else:
                    for label in prediction["labels"]:
                        boxes = torch.cat(
                            (
                                boxes,
                                torch.tensor(
                                    [
                                        label["box2d"]["x1"],
                                        label["box2d"]["y1"],
                                        label["box2d"]["x2"],
                                        label["box2d"]["y2"],
                                        label["score"],
                                    ],
                                ).unsqueeze(0),
                            ),
                            dim=0,
                        )
                        idx_to_class_mapping = MetadataCatalog.get(
                            "bdd100k_sample_val"
                        ).idx_to_class_mapping
                        class_to_idx_mapping = {
                            v: k for k, v in idx_to_class_mapping.items()
                        }
                        class_ids = torch.cat(
                            (
                                class_ids,
                                torch.tensor(
                                    [class_to_idx_mapping[label["category"]]]
                                ),
                            )
                        )

                    self.search_dict[video_name][frame_index] = Boxes2D(
                        boxes, class_ids
                    )

        assert len(batch_inputs) == 1, "Currently only BS=1 supported!"
        frame_id = batch_inputs[0].metadata.frame_index
        # init graph at begin of sequence
        if frame_id == 0:
            self.track_graph.reset()

        # using given detections
        image = batch_inputs[0].image
        video_name = batch_inputs[0].metadata.video_name
        frame_index = batch_inputs[0].metadata.frame_index
        assert video_name in self.search_dict
        assert frame_index in self.search_dict[video_name]
        detections = [self.search_dict[video_name][frame_index]]

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
            tracks = Boxes2D(torch.empty(0, 5), torch.empty(0), torch.empty(0))
        else:
            det_features = self.feature_extractor(
                detections[0].boxes[:, :-1], image.tensor
            )
            tracks = self.track_graph(detections[0], frame_id, det_features)
        return dict(detect=detections, track=[tracks])  # type:ignore


class FeatureExtractor(object):
    def __init__(
        self, feature_net, use_cuda: bool = True, model_weight_path: str = None
    ):
        self.net = feature_net
        self.device = (
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        if model_weight_path:
            state_dict = torch.load(
                model_weight_path, map_location=lambda storage, loc: storage
            )["net_dict"]
            self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def __call__(self, instance_boxes: torch.Tensor, image: torch.Tensor):
        """im_crops shape: 3xHxW"""
        instance_images = roi_align(
            image, [instance_boxes], (64, 128), aligned=True
        )
        instance_images = self.norm(instance_images)
        with torch.no_grad():
            features = self.net(instance_images, train=False)
        return features
