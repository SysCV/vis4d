"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from sort_graph import SORTTrackGraph
from sort_model import SORT

from openmt import config
from openmt.config import DataloaderConfig as Dataloader
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
            images_per_gpu=2,
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
        ),
        train=[
            config.Dataset(
                name="bdd100k_sample_train",
                type="scalabel",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/track/track/bdd100k-samples/images/",
                config_path="openmt/engine/testcases/track/bdd100k-samples/"
                "config.toml",
            )
        ],
        test=[
            config.Dataset(
                name="bdd100k_sample_val",
                type="scalabel",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/engine/testcases/track/bdd100k-samples/"
                "images/",
                config_path="openmt/engine/testcases/track/bdd100k-samples/"
                "config.toml",
            )
        ],
    )

    # TODO choose according to setup, add pretrained weights if necessary
    # CPU
    test(conf)

    # single GPU
    conf.launch = config.Launch(device="cuda")
    test(conf)
