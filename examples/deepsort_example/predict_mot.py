"""Example for dynamic api usage with SORT."""
# import the SORT components, needs to be imported to be registered
from examples.deepsort_example.deepsort_graph import DeepSORTTrackGraph
from examples.deepsort_example.deepsort_model import DeepSORT

from vist import config
from vist.config import DataloaderConfig as Dataloader
from vist.config import Augmentation
from vist.engine import train, test, predict
from vist.model import BaseModelConfig

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
        num_instances=625,  # 517 for MOT16 train, # 625 for using given pretrained weights
        prediction_path="weight/MOT16_det_feat",
    )

    conf = config.Config(
        model=BaseModelConfig(**deepsort_cfg),
        solver=config.Solver(
            images_per_gpu=1,  # 32
            lr_policy="WarmupMultiStepLR",
            base_lr=0.001,
            steps=[20000],
            max_iters=25000,
            log_period=100,
            checkpoint_period=1000,
            eval_period=25000,
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
        test=[
            config.Dataset(
                name="MOT16_test",
                type="MOTChallenge",
                annotations="data/MOT16/test",
                data_root="data/MOT16/test",
            )
        ],
    )

    conf.launch.weights = "/home/yinjiang/systm/openmt-workspace/DeepSORT/2021-06-25_12:21:34/model_final.pth"
    # conf.launch.weights = "/home/yinjiang/systm/examples/deepsort_example/checkpoint/original_ckpt.pth"
    conf.launch.device = "cuda"
    # import os
    # import shutil

    # if os.path.exists("visualization/"):
    #     shutil.rmtree("visualization/")
    # os.mkdir("visualization/")

    predict(conf)
