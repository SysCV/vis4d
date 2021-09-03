"""Example for dynamic api usage."""
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.models.detection import retinanet  # type: ignore

import vist.data.datasets.base
from vist import config
from vist.data.datasets import DataloaderConfig as Dataloader
from vist.engine.trainer import train
from vist.model import BaseModelConfig
from vist.model.detect import BaseDetector
from vist.model.optimize import BaseOptimizerConfig
from vist.struct import Boxes2D, Images, InputSample, LossesType, ModelOutput


class MyDetectorConfig(BaseModelConfig, extra="allow"):
    """My detector config."""

    abc: str


class MyDetector(BaseDetector):
    """Example detector model."""

    def __init__(self, cfg: BaseModelConfig) -> None:
        """Init detector."""
        super().__init__(cfg)
        self.cfg = MyDetectorConfig(**cfg.dict())  # type: MyDetectorConfig
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
        self, batch_inputs: List[List[InputSample]]
    ) -> ModelOutput:
        """Forward pass during testing stage.

        Args:
            batch_inputs: Model input (batched).

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
        model=dict(
            type="MyDetector",
            category_mapping={
                "pedestrian": 0,
                "rider": 1,
                "car": 2,
                "truck": 3,
                "bus": 4,
                "train": 5,
                "motorcycle": 6,
                "bicycle": 7,
            },
            image_channel_mode="RGB",
            optimizer=BaseOptimizerConfig(lr=0.001),
            abc="example_attribute",
        ),
        launch=config.Launch(samples_per_gpu=2, workers_per_gpu=0),
        train=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_sample_train",
                type="BDD100K",
                annotations="vist/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="vist/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="box_track",
                eval_metrics=["detect"],
                dataloader=Dataloader(skip_empty_samples=True),
            )
        ],
        test=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_sample_val",
                type="BDD100K",
                annotations="vist/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="vist/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="box_track",
                eval_metrics=["detect"],
            )
        ],
    )

    # choose according to setup
    # CPU
    train(conf)

    # single GPU
    trainer_args = {"gpus": "0,"}  # add arguments for PyTorchLightning trainer
    train(conf, trainer_args)

    # multi GPU
    trainer_args = {"gpus": "0,1"}
    train(conf, trainer_args)
