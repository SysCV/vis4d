"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
import vist.data.datasets.base
from vist import config
from vist.data.dataset_mapper import DataloaderConfig as Dataloader
from vist.engine import test
from vist.model import BaseModelConfig
from vist.model.sort_example.sort_graph import SORTTrackGraph
from vist.model.sort_example.sort_model import SORT

# Disable pylint for this file due to high overlap with detector example
# pylint: skip-file
if __name__ == "__main__":
    sort_detector_cfg = dict(
        type="D2TwoStageDetector",
        model_base="faster-rcnn/r50-fpn",
        num_classes=8,
    )
    sort_trackgraph_cfg = dict(type="SORTTrackGraph")
    sort_cfg = dict(
        type="SORT",
        detection=sort_detector_cfg,
        track_graph=sort_trackgraph_cfg,
        dataset="BDD100K",
        prediction_path="/home/yinjiang/systm/given_predictions/track_predictions.json",
    )

    conf = config.Config(
        model=BaseModelConfig(**sort_cfg),
        solver=config.Solver(
            images_per_gpu=32,
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            max_iters=100,
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
                # data_root="data/bdd100k/images/track/val/",
                # annotations="data/bdd100k/labels/box_track_20/val/",
                # data_root="vist/engine/testcases/track/bdd100k-samples/"
                # "images/",
                # annotations="vist/engine/testcases/track/bdd100k-samples/"
                # "labels",
                data_root="data/one_sequence/images/",
                annotations="data/one_sequence/labels",
                config_path="box_track",
                eval_metrics=["track"],
            )
        ],
    )

    # conf.launch.weights = "weight/model_0000199.pth"
    conf.launch.device = "cuda"
    conf.launch.num_gpus = 1
    test(conf)
