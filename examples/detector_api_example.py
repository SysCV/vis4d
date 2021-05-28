"""Example for dynamic api usage."""
from typing import List, Optional

import torch
import torchvision.models.detection.retinanet as retinanet  # type: ignore
from detectron2.engine import launch

from openmt import config
from openmt.config import DataloaderConfig as Dataloader
from openmt.engine import train
from openmt.model.detect import BaseDetector, BaseDetectorConfig
from openmt.struct import Boxes2D, DetectionOutput, Images, InputSample


class MyDetectorConfig(BaseDetectorConfig, extra="allow"):
    """My detector config."""

    abc: str


class MyDetector(BaseDetector):
    """Example detection module."""

    def __init__(self, cfg: BaseDetectorConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = MyDetectorConfig(**cfg.dict())
        self.retinanet = retinanet.retinanet_resnet50_fpn(pretrained=True)

    @property
    def device(self) -> torch.device:
        """Get device where detect input should be moved to."""
        raise NotImplementedError

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Normalize, pad and batch the input images."""
        raise NotImplementedError

    def forward(
        self,
        inputs: List[InputSample],
        targets: Optional[List[Boxes2D]] = None,
    ) -> DetectionOutput:
        """Detector forward function.

        Return backbone output features, proposals, detections and optionally
        training losses.
        """
        raise NotImplementedError


if __name__ == "__main__":
    my_detector_cfg = dict(type="MyDetector", abc="example_attribute")

    conf = config.Config(
        model=dict(
            type="DetectorWrapper",
            detection=BaseDetectorConfig(**my_detector_cfg),
        ),
        solver=config.Solver(
            images_per_gpu=2,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=100,
            eval_metrics=["detect"],
        ),
        dataloader=Dataloader(
            workers_per_gpu=0,
            ref_sampling_cfg=dict(type="uniform", scope=1, num_ref_imgs=0),
            categories=[
                "pedestrian",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ],
            remove_samples_without_labels=True,
        ),
        train=[
            config.Dataset(
                name="bdd100k_sample_train",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="box_track",
            )
        ],
        test=[
            config.Dataset(
                name="bdd100k_sample_val",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="box_track",
            )
        ],
    )

    # choose according to setup
    # CPU
    train(conf)

    # single GPU
    conf.launch = config.Launch(device="cuda")
    train(conf)

    # multi GPU
    conf.launch = config.Launch(device="cuda", num_gpus=2)
    launch(
        train,
        conf.launch.num_gpus,
        num_machines=conf.launch.num_machines,
        machine_rank=conf.launch.machine_rank,
        dist_url=conf.launch.dist_url,
        args=(conf,),
    )
