"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from examples.deepsort_example.deepsort_graph import DeepSORTTrackGraph
from examples.deepsort_example.deepsort_model import DeepSORT

import openmt.data.datasets.base
from vist import config
from vist.data.dataset_mapper import DataloaderConfig as Dataloader
from vist.data.transforms.base import AugmentationConfig as Augmentation
from vist.engine import train
from vist.model import BaseModelConfig

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
        num_instances=108524,
        prediction_path="weight/predictions.json",
    )

    conf = config.Config(
        model=BaseModelConfig(**deepsort_cfg),
        solver=config.Solver(
            images_per_gpu=32,  # 32 for train
            lr_policy="WarmupMultiStepLR",
            base_lr=0.0006,
            # steps=[30000, 40000],
            max_iters=50000,
            log_period=100,
            checkpoint_period=1000,
            eval_period=10000,
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
            compute_global_instance_ids=True,
            # train_augmentations=[
            #     Augmentation(type="Resize", kwargs={"shape": [720, 1280]}),
            #     Augmentation(type="RandomFlip", kwargs={"prob": 0.5}),
            # ],
            image_channel_mode="BGR",
        ),
        train=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_train",
                type="BDD100K",
                # annotations="vist/engine/testcases/track/bdd100k-samples/"
                # "labels",
                # data_root="vist/engine/testcases/track/bdd100k-samples/"
                # "images/",
                # annotations="data/one_sequence/labels",
                # data_root="data/one_sequence/images/",
                annotations="data/bdd100k/labels/box_track_20/train/",
                data_root="data/bdd100k/images/track/train/",
                config_path="box_track",
            )
        ],
        test=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_val",
                type="BDD100K",
                # annotations="vist/engine/testcases/track/bdd100k-samples/"
                # "labels",
                # data_root="vist/engine/testcases/track/bdd100k-samples/"
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
    # conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-26_10:59:19/model_0014999.pth"
    # conf.launch.weights = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.pth"
    # conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-26_20:30:08/model_0032999.pth"
    conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-28_11:38:49/model_0007999.pth"
    conf.launch.device = "cuda"
    conf.launch.num_gpus = 6
    conf.launch.resume = True
    # import os
    # import shutil
    # if os.path.exists("visualsization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")

    train(conf)
