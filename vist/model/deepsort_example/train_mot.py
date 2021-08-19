"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
import vist.data.datasets.base
from vist import config
from vist.data.dataset_mapper import DataloaderConfig as Dataloader
from vist.data.transforms.base import AugmentationConfig as Augmentation
from vist.engine import test, train
from vist.model import BaseModelConfig

from .deepsort_graph import DeepSORTTrackGraph
from .deepsort_model import DeepSORT

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
        max_boxes_num=512,
        dataset="MOT16",
        num_instances=517,  # 517 for MOT16 train, # 625 for using given pretrained weights
        prediction_path="weight/MOT16_det_feat",
    )

    conf = config.Config(
        model=BaseModelConfig(**deepsort_cfg),
        solver=config.Solver(
            images_per_gpu=2,  # 32 for train, # 1 for test on biwidl302
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            # steps=[20000],
            max_iters=10000,
            log_period=100,
            checkpoint_period=1000,
            eval_period=5000,
            eval_metrics=["track"],
        ),
        dataloader=Dataloader(
            workers_per_gpu=8,
            ref_sampling_cfg=dict(type="uniform", scope=1, num_ref_imgs=0),
            categories=[
                "pedestrian",
            ],
            remove_samples_without_labels=True,
            inference_sampling="sequence_based",
            compute_global_instance_ids=True,
            train_augmentations=[
                Augmentation(type="Resize", kwargs={"shape": [720, 1280]}),
                Augmentation(type="RandomFlip", kwargs={"prob": 0.5}),
            ],
        ),
        train=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="MOT16_train",
                type="MOTChallenge",
                annotations="data/MOT16/train",
                data_root="data/MOT16/train",
            )
        ],
        test=[
            vist.data.datasets.base.BaseDatasetConfig(
                name="MOT16_val",
                type="MOTChallenge",
                annotations="data/MOT16/train",
                data_root="data/MOT16/train",
            )
        ],
    )

    conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-25_12:21:34/model_final.pth"
    # conf.launch.weights = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.pth"
    conf.launch.device = "cuda"
    conf.launch.num_gpus = 1
    # import os
    # import shutil

    # if os.path.exists("visualization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")

    train(conf)
    # test(conf)
