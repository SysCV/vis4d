"""QDTrack BDD100K inference example."""
from __future__ import annotations

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    LoggingCallback,
)
from vis4d.config.default.data.dataloader import default_image_dataloader
from vis4d.model.track.qdtrack import FasterRCNNQDTrack
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as CK
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import (
    DataConnectionInfo,
    StaticDataConnector,
    data_key,
    pred_key,
)

from vis4d.eval.track.scalabel import ScalabelEvaluator

from vis4d.data.loader import VideoDataPipe
from vis4d.config.default.runtime import set_output_dir

CONN_BBOX_2D_TEST = {
    CK.images: CK.images,
    CK.input_hw: "images_hw",
    CK.frame_ids: CK.frame_ids,
}

CONN_BDD100K_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "data_names": data_key("name"),
    "video_names": data_key("videoName"),
    "boxes_list": pred_key("boxes"),
    "class_ids_list": pred_key("class_ids"),
    "scores_list": pred_key("scores"),
    "track_ids_list": pred_key("track_ids"),
}


def get_config() -> ConfigDict:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()
    config.n_gpus = 8
    config.work_dir = "vis4d-workspace"
    config.experiment_name = "qdtrack_bdd100k"
    config = set_output_dir(config)

    ckpt_path = (
        "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
    )

    # Hyper Parameters
    params = ConfigDict()
    params.samples_per_gpu = 4
    params.lr = 0.01
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = ConfigDict()
    dataset_root = "data/bdd100k/images/track/val/"
    annotation_path = "data/bdd100k/labels/box_track_20/val/"
    config_path = "box_track"
    data_backend = HDF5Backend()

    # TODO: Add train dataset
    train_dataset_cfg = None
    train_dataloader_cfg = None
    data.train_dataloader = {"bdd100k_train": train_dataloader_cfg}

    # Test
    test_dataset_cfg = class_config(
        BDD100K,
        data_root=dataset_root,
        targets_to_load=(),
        annotation_path=annotation_path,
        config_path=config_path,
        data_backend=data_backend,
    )

    test_preprocess_cfg = class_config(
        "vis4d.data.transforms.compose",
        transforms=[
            class_config(
                "vis4d.data.transforms.resize.resize_image",
                shape=(720, 1280),
                keep_ratio=True,
                align_long_edge=True,
            ),
            class_config(
                "vis4d.data.transforms.normalize.normalize_image",
            ),
        ],
    )

    test_batchprocess_cfg = class_config(
        "vis4d.data.transforms.compose",
        transforms=[
            class_config(
                "vis4d.data.transforms.pad.pad_image",
            ),
        ],
    )

    test_dataloader_cfg = default_image_dataloader(
        test_preprocess_cfg,
        test_dataset_cfg,
        num_samples_per_gpu=1,
        batchprocess_cfg=test_batchprocess_cfg,
        data_pipe=VideoDataPipe,
        train=False,
    )
    data.test_dataloader = {"bdd100k_eval": test_dataloader_cfg}
    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    num_classes = len(bdd100k_track_map)

    config.model = class_config(
        FasterRCNNQDTrack,
        num_classes=num_classes,
        weights=ckpt_path,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    config.loss = None  # TODO: implement loss

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        optimizer_cfg(
            optimizer=class_config(optim.SGD, lr=params.lr),
            lr_scheduler=class_config(
                MultiStepLR, milestones=[8, 11], gamma=0.1
            ),
            lr_warmup=None,
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    # This defines how the output of each component is connected to the next
    # component. This is a very important part of the config. It defines the
    # data flow of the pipeline.
    # We use the default connections provided for faster_rcnn.
    config.data_connector = class_config(
        StaticDataConnector,
        connections=DataConnectionInfo(
            test=CONN_BBOX_2D_TEST,
            callbacks={"bdd100k_eval_test": CONN_BDD100K_EVAL},
        ),
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################
    # Here we define the evaluator. We need to define the connections
    # between the evaluator and the data connector in the data connector
    # section. And use the same name here.
    eval_callbacks = {
        "bdd100k_eval": class_config(
            EvaluatorCallback,
            save_prefix=config.output_dir,
            evaluator=class_config(
                ScalabelEvaluator,
                annotation_path=annotation_path,
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        ),
    }

    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################
    # Here we define general, all purpose callbacks. Note, that these callbacks
    # do not need to be registered with the data connector.
    logger_callback = {
        "logger": class_config(LoggingCallback, refresh_rate=50)
    }
    ckpt_callback = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.output_dir,
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    # Assign the defined callbacks to the config
    config.shared_callbacks = {**logger_callback, **eval_callbacks}

    config.train_callbacks = {**ckpt_callback}

    config.test_callbacks = {}

    ######################################################
    ##                  PL CALLBACKS                    ##
    ######################################################
    pl_trainer = ConfigDict()

    pl_callbacks: list[pl.callbacks.Callback] = []

    config.pl_trainer = pl_trainer
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
