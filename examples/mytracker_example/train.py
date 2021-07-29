"""."""
# needs to be imported to be registered
from examples.mytracker_example.mytracker import MyTracker

import openmt.data.datasets.base
from openmt import config
from openmt.data.dataset_mapper import DataloaderConfig as Dataloader
from openmt.engine import train, test
from openmt.model import BaseModelConfig

# Disable pylint for this file due to high overlap with detector example
# pylint: skip-file
if __name__ == "__main__":
    qdtrack_graph_cfg = dict(type="QDTrackGraph", keep_in_memory=10)
    losses_cfg = [
        openmt.model.track.losses.base.LossConfig(
            type="MultiPosCrossEntropyLoss", loss_weight=0.25
        ),
        openmt.model.track.losses.base.LossConfig(
            type="EmbeddingDistanceLoss",
            loss_weight=1.0,
            neg_pos_ub=3,
            pos_margin=0,
            neg_margin=0.3,
            hard_mining=True,
        ),
    ]
    my_tracker_cfg = dict(
        type="MyTracker",
        track_graph=qdtrack_graph_cfg,
        losses=losses_cfg,
        softmax_temp=-1.0,
        embedding_dim=512,
        num_classes=8,
    )

    conf = config.Config(
        model=BaseModelConfig(**my_tracker_cfg),
        solver=config.Solver(
            images_per_gpu=1,  # 32 for train
            lr_policy="WarmupMultiStepLR",
            base_lr=0.0006,
            # steps=[30000, 40000],
            max_iters=50000,
            log_period=10,
            checkpoint_period=1000,
            eval_period=10000,
        ),
        dataloader=Dataloader(
            workers_per_gpu=8,
            ref_sampling_cfg=dict(
                type="uniform",
                scope=2,
                num_ref_imgs=1,
                skip_nomatch_samples=True,
            ),
            # ref_sampling_cfg=dict(type="uniform", scope=1, num_ref_imgs=0),
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
            image_channel_mode="BGR",
            # train_augmentations=[
            #     Augmentation(type="Resize", kwargs={"shape": [720, 1280]}),
            #     Augmentation(type="RandomFlip", kwargs={"prob": 0.5}),
            # ],
        ),
        train=[
            openmt.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_train",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                # annotations="data/one_sequence/labels",
                # data_root="data/one_sequence/images/",
                # annotations="data/bdd100k/labels/box_track_20/train/",
                # data_root="data/bdd100k/images/track/train/",
                config_path="box_track",
            )
        ],
        test=[
            openmt.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_val",
                type="BDD100K",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                # annotations="data/one_sequence/labels",
                # data_root="data/one_sequence/images/",
                # annotations="data/bdd100k/labels/box_track_20/val/",
                # data_root="data/bdd100k/images/track/val/",
                config_path="box_track",
                eval_metrics=["detect", "track"],
            )
        ],
    )

    # choose according to setup
    conf.launch.weights = (
        "/home/yinjiang/systm/examples/checkpoint/weight_of_6frames.pth"
    )
    conf.launch.device = "cpu"
    # conf.launch.num_gpus = 1
    # conf.launch.resume = True
    # import os
    # import shutil
    # if os.path.exists("visualsization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")
    train(conf)
    # test(conf)
