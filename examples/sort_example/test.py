"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from sort_graph import SORTTrackGraph
from sort_model import SORT

import vist.data.datasets.base
from vist import config
from vist.data.datasets import DataloaderConfig as Dataloader
from vist.engine.trainer import test
from vist.model import BaseModelConfig
from vist.model.optimize import BaseOptimizerConfig

# Disable pylint for this file due to high overlap with detector example
# pylint: skip-file
if __name__ == "__main__":
    sort_detector_cfg = dict(
        type="D2TwoStageDetector",
        model_base="faster-rcnn/r50-fpn",
    )
    sort_trackgraph_cfg = dict(type="SORTTrackGraph")
    sort_cfg = dict(
        type="SORT",
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
        detection=sort_detector_cfg,
        track_graph=sort_trackgraph_cfg,
    )

    conf = config.Config(
        model=BaseModelConfig(**sort_cfg),
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

    # TODO choose according to setup, add pretrained weights if necessary
    # conf.launch.weights = "/path/to/weight.ckpt"
    # CPU
    test(conf)

    # single GPU
    trainer_args = {"gpus": "0,"}  # add arguments for PyTorchLightning trainer
    test(conf, trainer_args)
