"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from sort_graph import SORTTrackGraph
from sort_model import SORT

import openmt.data.datasets.base
from openmt import config

from openmt.config import Augmentation
from openmt.data.build import DataloaderConfig as Dataloader
from openmt.engine import test
from openmt.model import BaseModelConfig

# Disable pylint for this file due to high overlap with detector example
# pylint: skip-file
if __name__ == "__main__":
    sort_detector_cfg = dict(
        type="D2GeneralizedRCNN",
        model_base="faster-rcnn/r50-fpn",
        num_classes=8,
    )
    sort_trackgraph_cfg = dict(type="SORTTrackGraph")
    sort_cfg = dict(
        type="SORT",
        detection=sort_detector_cfg,
        track_graph=sort_trackgraph_cfg,
    )

    conf = config.Config(
        model=BaseModelConfig(**sort_cfg),
        solver=config.Solver(
            images_per_gpu=32,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=100,
            eval_metrics=["detect", "track"],
        ),
        dataloader=Dataloader(
            workers_per_gpu=8,
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
            inference_sampling="sequence_based",
            train_augmentations=[
                Augmentation(type="Resize", kwargs={"shape": [720, 1280]}),
                Augmentation(type="RandomFlip", kwargs={"prob": 0.5}),
            ],
            test_augmentations=[
                Augmentation(type="Resize", kwargs={"shape": [720, 1280]})
            ],
        ),
        train=[
            openmt.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_sample_train",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/track/track/bdd100k-samples/images/",
                config_path="box_track",
            )
        ],
        test=[
            openmt.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_sample_val",
                type="BDD100K",
                # annotations="openmt/engine/testcases/track/bdd100k-samples/"
                # "labels",
                # annotations="data/bdd100k/labels/box_track_20/val/",
                annotations="data/one_sequence/labels",
                # data_root="openmt/engine/testcases/track/bdd100k-samples/"
                # "images/",
                # data_root="data/bdd100k/images/track/val/",
                data_root="data/one_sequence/images/",
                config_path="box_track",
            )
        ],
    )

    # TODO choose according to setup, add pretrained weights if necessary
    # CPU
    conf.launch.weights = "weight/model_0000199.pth"
    # import os
    # import shutil
    # if os.path.exists("visualization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")

    test(conf)

    # single GPU
    # conf.launch = config.Launch(device="cuda")
    # test(conf)
