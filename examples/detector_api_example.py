"""Example for dynamic api usage."""
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.models.detection.retinanet as retinanet  # type: ignore

from detectron2.engine import launch
from openmt import config, detect
from openmt.data import DataloaderConfig as Dataloader
from openmt.model.detect import BaseDetector, BaseDetectorConfig
from openmt.struct import Boxes2D, DetectionOutput, ImageList


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

    def preprocess_image(
        self, batched_inputs: Tuple[Dict[str, torch.Tensor]]
    ) -> ImageList:
        """Normalize, pad and batch the input images."""
        raise NotImplementedError

    def forward(
        self,
        inputs: ImageList,
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
            images_per_batch=2,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=100,
        ),
        dataloader=Dataloader(
            num_workers=0,
            sampling_cfg=dict(type="uniform", scope=1, num_ref_imgs=0),
        ),
        train=[
            config.Dataset(
                name="bdd100k_sample_train",
                type="coco",
                annotations="openmt/detect/testcases/bdd100k-samples/"
                            "annotation_coco.json",
                data_root="openmt/detect/testcases/bdd100k-samples/images/",
            )
        ],
        test=[
            config.Dataset(
                name="bdd100k_sample_val",
                type="coco",
                annotations="openmt/detect/testcases/bdd100k-samples/"
                            "annotation_coco.json",
                data_root="openmt/detect/testcases/bdd100k-samples/images/",
            )
        ],
    )

    # choose according to setup
    # CPU
    detect.train(conf)

    # single GPU
    conf.launch = config.Launch(device='cuda')
    detect.train(conf)

    # multi GPU
    conf.launch = config.Launch(device='cuda', num_gpus=2)
    launch(
        detect.train,
        conf.launch.num_gpus,
        num_machines=conf.launch.num_machines,
        machine_rank=conf.launch.machine_rank,
        dist_url=conf.launch.dist_url,
        args=(conf,),
    )
