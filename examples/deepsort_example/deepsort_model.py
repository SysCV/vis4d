"""deep SORT model definition."""
from typing import List, Dict, Union
import json
import torch
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
    max_boxes_num: int = 512
    num_instances: int = 108524  # 108524 # 625
    # featurenet_weight_path: Union[
    #     str, None
    # ] = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.t7"
    featurenet_weight_path: Union[str, None] = None
    # featurenet_weight_path: Union[
    #     str, None
    # ] = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-01_08:40:48/model_0006999.pth"


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert torch.cuda.is_available(), "cuda is not available"
        self.feature_net.to(self.device)
        if self.cfg.featurenet_weight_path:
            state_dict = torch.load(
                self.cfg.featurenet_weight_path,
            )["model_state_dict"]
            self.feature_net.load_state_dict(state_dict)

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
        instance_boxes = torch.empty([0, 5])
        for batch_index, label in enumerate(labels):
            boxes = label.boxes[:, :-1].float()  # type: ignore
            batch_column = torch.ones(len(boxes), 1) * batch_index
            batch_boxes = torch.cat((batch_column, boxes), dim=1)
            instance_boxes = torch.cat((instance_boxes, batch_boxes), dim=0)
        instance_boxes = instance_boxes.to(self.device)
        instance_ids = torch.cat([label.track_ids for label in labels], dim=0).to(self.device)  # type: ignore
        batch_size = min(self.cfg.max_boxes_num, len(instance_boxes))
        # print("len(instance_boxes):   ", len(instance_boxes))
        # print("batch_size:   ", batch_size)
        indices = torch.randperm(len(instance_boxes))[:batch_size]
        instance_boxes = instance_boxes[indices]
        instance_ids = instance_ids[indices]

        instance_images = roi_align(
            inputs_images, instance_boxes, (64, 128), aligned=True
        )

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
        detections = [
            self.search_dict[video_name][frame_index].to(self.device)
        ]

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
            instance_images = roi_align(
                image_tensor, [instance_boxes], (64, 128), aligned=True
            )
            det_features = self.feature_net(instance_images, train=False)
            tracks = self.track_graph(detections[0], frame_id, det_features)
        from openmt.vis.image import imsave_bboxes

        # imsave_bboxes(
        #     image.tensor[0],
        #     detections,
        #     frame_id,
        # )
        return dict(detect=detections, track=[tracks])  # type:ignore
