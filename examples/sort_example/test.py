"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from sort_graph import SORTTrackGraph
from sort_model import SORT

import vist.data.datasets.base
from vist import config
from vist.data.dataset import DataloaderConfig as Dataloader
from vist.engine import test
from vist.model import BaseModelConfig

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
        detection=sort_detector_cfg,
        track_graph=sort_trackgraph_cfg,
    )

    conf = config.Config(
        model=BaseModelConfig(**sort_cfg),
        solver=config.Solver(
            samples_per_gpu=2,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=100,
            eval_metrics=["track"],
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
            image_channel_mode="BGR",
        ),
        train=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="bdd100k_sample_train",
                type="BDD100K",
                annotations="vist/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="vist/track/track/bdd100k-samples/images/",
                config_path="box_track",
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
            )
        ],
    )

    # TODO choose according to setup, add pretrained weights if necessary
    # CPU
    test(conf)

    # single GPU
    conf.launch = config.Launch(device="cuda")
    test(conf)
