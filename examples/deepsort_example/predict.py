"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from deepsort_graph import DeepSORTTrackGraph
from deepsort_model import DeepSORT

from openmt import config
from openmt.config import DataloaderConfig as Dataloader
from openmt.engine import predict
from openmt.model import BaseModelConfig


# Disable pylint for this file due to high overlap with detector example
# pylint: skip-file
if __name__ == "__main__":
    deepsort_detector_cfg = dict(  # TODO load pretrained weights
        type="D2GeneralizedRCNN",
        model_base="faster-rcnn/r50-fpn",
        num_classes=8,
    )
    deepsort_trackgraph_cfg = dict(type="DeepSORTTrackGraph")
    deepsort_cfg = dict(
        type="DeepSORT",
        detection=deepsort_detector_cfg,
        track_graph=deepsort_trackgraph_cfg,
    )

    conf = config.Config(
        model=BaseModelConfig(**deepsort_cfg),
        solver=config.Solver(
            images_per_gpu=2,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=100,
            eval_metrics=["detect", "track"],
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
            inference_sampling="sequence_based",
        ),
        train=[
            config.Dataset(
                name="bdd100k_sample_train",
                type="scalabel",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/track/track/bdd100k-samples/images/",
                ignore=["other person", "other vehicle", "trailer"],
                name_mapping={
                    "bike": "bicycle",
                    "caravan": "car",
                    "motor": "motorcycle",
                    "person": "pedestrian",
                    "van": "car",
                },
            )
        ],
        test=[
            config.Dataset(
                name="bdd100k_sample_val",
                type="scalabel",
                # annotations="openmt/engine/testcases/track/bdd100k-samples/"
                # "labels",
                # annotations="data/bdd100k/labels/box_track_20/val/",
                annotations="data/one_sequence/labels",
                # data_root="openmt/engine/testcases/track/bdd100k-samples/"
                # "images/",
                # data_root="data/bdd100k/images/track/val/",
                data_root="data/one_sequence/images/",
            )
        ],
    )

    # choose according to setup
    # CPU
    # conf.launch = config.Launch(weights="weight/model_0000199.pth")
    # import os
    # import shutil
    # if os.path.exists("visualization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")
    predict(conf)

    # single GPU
    # conf.launch = config.Launch(device='cuda')
    # predict(conf)
