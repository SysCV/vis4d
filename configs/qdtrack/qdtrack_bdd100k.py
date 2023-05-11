"""QDTrack BDD100K inference example."""
from __future__ import annotations

import pytorch_lightning as pl

from vis4d.common.callbacks import EvaluatorCallback
from vis4d.config.default.dataloader import get_dataloader_config
from vis4d.config.default.runtime import (
    get_generic_callback_config,
    get_pl_trainer_args,
    set_output_dir,
)
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.loader import VideoDataPipe
from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.normalize import NormalizeImage
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeBoxes2D,
    ResizeImage,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.connectors import DataConnector, data_key, pred_key
from vis4d.eval.bdd100k import BDD100KTrackingEvaluator
from vis4d.model.track.qdtrack import FasterRCNNQDTrack

CONN_BBOX_2D_TEST = {
    K.images: K.images,
    K.input_hw: "images_hw",
    K.frame_ids: K.frame_ids,
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
    config.work_dir = "vis4d-workspace"
    config.experiment_name = "qdtrack_bdd100k"
    config = set_output_dir(config)

    ckpt_path = (
        "https://dl.cv.ethz.ch/vis4d/qdtrack_bdd100k_frcnn_res50_heavy_augs.pt"
    )

    # Hyper Parameters
    params = ConfigDict()
    params.samples_per_gpu = 4
    params.workers_per_gpu = 4
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
    data_backend = class_config(HDF5Backend)

    # TODO: Add train dataset
    data.train_dataloader = None

    # Test
    test_dataset_cfg = class_config(
        BDD100K,
        data_root=dataset_root,
        keys_to_load=(K.images),
        annotation_path=annotation_path,
        config_path=config_path,
        data_backend=data_backend,
    )

    preprocess_transforms = [
        class_config(
            GenerateResizeParameters,
            shape=(720, 1280),
            keep_ratio=True,
        ),
        class_config(ResizeImage),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(class_config(NormalizeImage))

    test_preprocess_cfg = class_config(
        compose,
        transforms=preprocess_transforms,
    )

    test_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(PadImages),
            class_config(ToTensor),
        ],
    )

    data.test_dataloader = get_dataloader_config(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        data_pipe=VideoDataPipe,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=1,
        workers_per_gpu=params.workers_per_gpu,
        train=False,
    )

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
    config.optimizers = None  # TODO: implement optimizer

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.data_connector = class_config(
        DataConnector,
        test=CONN_BBOX_2D_TEST,
        callbacks={"bdd100k_eval_test": CONN_BDD100K_EVAL},
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################
    eval_callbacks = {
        "bdd100k_eval": class_config(
            EvaluatorCallback,
            save_prefix=config.output_dir,
            evaluator=class_config(
                BDD100KTrackingEvaluator,
                annotation_path=annotation_path,
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        ),
    }

    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################
    # Generic callbacks
    logger_callback, ckpt_callback = get_generic_callback_config(
        config, params
    )

    # Assign the defined callbacks to the config
    config.shared_callbacks = {**logger_callback, **eval_callbacks}

    config.train_callbacks = {**ckpt_callback}

    config.test_callbacks = {}

    ######################################################
    ##                  PL CALLBACKS                    ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_pl_trainer_args()
    pl_trainer.max_epochs = params.num_epochs
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
