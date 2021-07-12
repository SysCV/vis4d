"""Example for dynamic api usage."""
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.engine import launch
from torchvision.models.detection import retinanet  # type: ignore

import openmt.data.datasets.base
from openmt import config
from openmt.data.dataset_mapper import DataloaderConfig as Dataloader
from openmt.engine import train
from openmt.model import BaseModelConfig
from openmt.model.detect import BaseDetector
from openmt.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput


class MyDetectorConfig(BaseModelConfig, extra="allow"):
    """My detector config."""

    abc: str


class MyDetector(BaseDetector):
    """Example detector model."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__()
        self.cfg = MyDetectorConfig(**cfg.dict())
        self.retinanet = retinanet.retinanet_resnet50_fpn(pretrained=True)

    def preprocess_image(self, batched_inputs: List[InputSample]) -> Images:
        """Normalize, pad and batch the input images."""
        raise NotImplementedError

    def forward_train(
        self, batch_inputs: List[List[InputSample]]
    ) -> LossesType:
        """Forward pass during training stage.

        Args:
            batch_inputs: Model input. Batched, including possible reference
            views.

        Returns:
            LossesType: A dict of scalar loss tensors.
        """
        raise NotImplementedError

    def forward_test(
        self, batch_inputs: List[InputSample], postprocess: bool = True
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Args:
            batch_inputs: Model input (batched).
            postprocess: If output should be postprocessed to original
            resolution.

        Returns:
            ModelOutput: Dict of LabelInstance results, e.g. tracking and
            separate models result.
        """
        raise NotImplementedError

    def extract_features(self, images: Images) -> Dict[str, torch.Tensor]:
        """Detector feature extraction stage.

        Return backbone output features
        """
        raise NotImplementedError

    def generate_detections(
        self,
        images: Images,
        features: Dict[str, torch.Tensor],
        proposals: List[Boxes2D],
        targets: Optional[List[Boxes2D]] = None,
        compute_detections: bool = True,
    ) -> Tuple[Optional[List[Boxes2D]], LossesType]:
        """Detector second stage (RoI Head).

        Return losses (empty if no targets) and optionally detections.
        """
        raise NotImplementedError


if __name__ == "__main__":
    conf = config.Config(
        model=dict(type="MyDetector", abc="example_attribute"),
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
            image_channel_mode="BGR",
        ),
        train=[
            openmt.data.datasets.base.BaseDatasetConfig(
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
            openmt.data.datasets.base.BaseDatasetConfig(
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
