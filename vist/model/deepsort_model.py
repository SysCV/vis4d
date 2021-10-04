"""deep SORT model definition."""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from vist.struct import Boxes2D, InputSample, LossesType, ModelOutput
from vist.common.bbox.poolers import RoIPoolerConfig, build_roi_pooler
from .base import BaseModel, BaseModelConfig
from .load_predictions import load_predictions
from .track.graph import TrackGraphConfig, build_track_graph


class BasicBlock(nn.Module):  # type: ignore
    """Basic build block."""

    def __init__(self, c_in: int, c_out: int, is_downsample: bool = False):
        """Init."""
        super().__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            c_out, c_out, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out),
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.is_downsample = True

    def forward(self, x: torch.Tensor):
        """Forward."""
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(
    c_in: int, c_out: int, repeat_times: int, is_downsample: bool = False
):
    """Make layers."""
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


class FeatureNet(nn.Module):  # type: ignore
    """Deep feature net."""

    def __init__(self, num_classes: int = 625):
        """Init."""
        super().__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(32, 32, repeat_times=2, is_downsample=False)
        # 32 64 32
        self.layer2 = make_layers(32, 64, repeat_times=2, is_downsample=True)
        # 64 32 16
        self.layer3 = make_layers(64, 128, repeat_times=2, is_downsample=True)
        # 128 16 8
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128 * 16 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
        )
        # 256 1 1
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self.norm = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        """Forward function of feature net.

        output size: N x 128
        """
        x = torch.div(x, 255.0)
        x = self.norm(x)
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        if not train:
            x = self.dense[0](x)
            x = self.dense[1](x)
            # x is normalized to a unit sphere
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        # N x 128
        # classifier
        x = self.classifier(x)
        return x


class DeepSORTConfig(BaseModelConfig, extra="allow"):
    """deep SORT config."""

    detection: BaseModelConfig
    track_graph: TrackGraphConfig
    max_boxes_num: int = 512
    dataset: str
    num_instances: int
    roi_align_config: RoIPoolerConfig
    prediction_path: str


class DeepSORT(BaseModel):
    """deep SORT tracking module."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__(cfg)
        self.cfg = DeepSORTConfig(**cfg.dict())
        self.track_graph = build_track_graph(self.cfg.track_graph)
        self.search_dict: Dict[str, Dict[int, Boxes2D]] = dict()
        self.feature_net = FeatureNet(num_classes=self.cfg.num_instances)
        self.roi_align = build_roi_pooler(self.cfg.roi_align_config)

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
        labels = []
        for inp in batch_inputs:
            assert inp[0].boxes2d is not None
            labels.append(inp[0].boxes2d.to(self.device))
        instance_ids = torch.cat([label.track_ids for label in labels], dim=0)
        instance_images = self.roi_align.pool([inputs_images], labels)
        batch_size = min(
            self.cfg.max_boxes_num, len(instance_images)  # type:ignore
        )
        indices = torch.randperm(len(instance_images))[:batch_size]
        instance_images = instance_images[indices]
        instance_ids = instance_ids[indices]

        cls_output = self.feature_net(instance_images, train=True)
        instance_ids = instance_ids.long()
        feature_net_loss = torch.nn.functional.cross_entropy(
            cls_output, instance_ids, reduction="mean"
        )
        return {"feature_net_loss": feature_net_loss}  # type: ignore

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
            image_tensor = image.tensor.to(self.device)
            instance_images = self.roi_align.pool(
                [image_tensor], [detections[0]]
            )
            det_features = self.feature_net(instance_images, train=False)
            tracks = self.track_graph(detections[0], frame_id, det_features)
        return dict(detect=detections, track=[tracks])  # type:ignore
