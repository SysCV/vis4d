"""Example for dynamic api usage with SORT."""
from openmt.model import BaseModelConfig
from openmt import config
from openmt.engine import predict
from openmt.data import DataloaderConfig as Dataloader

# import the SORT components
from sort_graph import SORTTrackGraph
from sort_model import SORT

if __name__ == "__main__":
    print("hi")
    sort_detector_cfg = dict(  # TODO load pretrained weights
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
            eval_metrics=["detect"],
        ),
        dataloader=Dataloader(
            workers_per_gpu=0,
            ref_sampling_cfg=dict(type="uniform", scope=1, num_ref_imgs=0),
        ),
        train=[
            config.Dataset(
                name="bdd100k_sample_train",
                type="scalabel",
                annotations="openmt/engine/testcases/track/bdd100k-samples/"
                "labels",
                data_root="openmt/track/track/bdd100k-samples/images/",
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
            )
        ],
    )

    # choose according to setup
    # CPU
    predict(conf)

    # single GPU
    # conf.launch = config.Launch(device='cuda')
    # predict(conf)
