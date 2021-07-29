"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from deepsort_graph import DeepSORTTrackGraph
from deepsort_model import DeepSORT

import openmt.data.datasets.base
from openmt import config
from openmt.data.dataset_mapper import DataloaderConfig as Dataloader
from openmt.data.transforms.base import AugmentationConfig as Augmentation

# from openmt.config import Augmentation
from openmt.engine import test
from openmt.model import BaseModelConfig

# Disable pylint for this file due to high overlap with detector example
# pylint: skip-file
if __name__ == "__main__":
    deepsort_detector_cfg = dict(  # TODO load pretrained weights
        type="D2GeneralizedRCNN",
        model_base="faster-rcnn/r50-fpn",
        num_classes=8,
    )
    deepsort_trackgraph_cfg = dict(
        type="DeepSORTTrackGraph", dataset="bdd100k_val"
    )
    deepsort_cfg = dict(
        type="DeepSORT",
        detection=deepsort_detector_cfg,
        track_graph=deepsort_trackgraph_cfg,
        max_boxes_num=512,
        featurenet_weight_path=None,
        dataset="BDD100K",
        num_instances=108524,  # 108524 for BDD100K train, # 625 for using given pretrained weights
        prediction_path="weight/predictions.json",
    )

    conf = config.Config(
        model=BaseModelConfig(**deepsort_cfg),
        solver=config.Solver(
            images_per_gpu=32,  # 32, # 2 in biwidl302
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=1000,
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
            image_channel_mode="BGR",
        ),
        train=[
            openmt.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_train",
                type="BDD100K",
                # annotations="openmt/engine/testcases/track/bdd100k-samples/"
                # "labels",
                # data_root="openmt/engine/testcases/track/bdd100k-samples/"
                # "images/",
                annotations="data/one_sequence/labels",
                data_root="data/one_sequence/images/",
                # annotations="data/bdd100k/labels/box_track_20/train/",
                # data_root="data/bdd100k/images/track/train/",
                config_path="box_track",
            )
        ],
        test=[
            openmt.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_val",
                type="BDD100K",
                # annotations="openmt/engine/testcases/track/bdd100k-samples/"
                # "labels",
                # data_root="openmt/engine/testcases/track/bdd100k-samples/"
                # "images/",
                # annotations="data/one_sequence/labels",
                # data_root="data/one_sequence/images/",
                annotations="data/bdd100k/labels/box_track_20/val/",
                data_root="data/bdd100k/images/track/val/",
                config_path="box_track",
                eval_metrics=["track"],
            )
        ],
    )

    # choose according to setup
    conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-28_21:24:04/model_0034999.pth"  # deepsort trained on BDD100K
    # conf.launch.weights = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.pth"
    conf.launch.device = "cuda"
    conf.launch.num_gpus = 4

    # import os
    # import shutil
    # if os.path.exists("visualization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")

    test(conf)
