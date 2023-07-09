# pylint: disable=duplicate-code
"""YOLOX COCO."""
from __future__ import annotations

import lightning.pytorch as pl

from vis4d.config import class_config
from vis4d.config.common.datasets.coco.yolox import (
    CONN_COCO_BBOX_EVAL,
    get_coco_yolox_cfg,
)
from vis4d.config.common.models.yolox import (
    get_yolox_cfg,
    get_yolox_optimizers_cfg,
)
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.default.data_connectors import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_VIS,
)
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.const import CommonKeys as K
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import (
    EvaluatorCallback,
    VisualizerCallback,
    YOLOXModeSwitchCallback,
    YOLOXSyncNormCallback,
)
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.eval.coco import COCODetectEvaluator
from vis4d.vis.image import BoundingBoxVisualizer

CONN_BBOX_2D_TRAIN = {"images": K.images}


def get_config() -> ExperimentConfig:
    """Returns the YOLOX config dict for the coco detection task.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="yolox_tiny_300e_coco")
    config.check_val_every_n_epoch = 10

    # High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 8
    params.workers_per_gpu = 4
    params.lr = 0.01
    params.num_epochs = 300
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/coco"
    train_split = "train2017"
    test_split = "val2017"

    data_backend = class_config(HDF5Backend)

    config.data = get_coco_yolox_cfg(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        scaling_ratio_range=(0.5, 1.5),
        resize_size=(320, 320),
        test_image_size=(416, 416),
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, config.loss = get_yolox_cfg(params.num_classes, "tiny")

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    steps_per_epoch, num_last_epochs, warmup_epochs = 1849, 15, 5
    config.optimizers = get_yolox_optimizers_cfg(
        params.lr,
        params.num_epochs,
        steps_per_epoch,
        warmup_epochs,
        num_last_epochs,
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(
        config.output_dir, refresh_rate=config.log_every_n_steps
    )

    # YOLOX callbacks
    callbacks += [
        class_config(
            YOLOXModeSwitchCallback,
            switch_epoch=params.num_epochs - num_last_epochs,
        ),
        class_config(YOLOXSyncNormCallback),
    ]

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBoxVisualizer, vis_freq=100, image_mode="BGR"
            ),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_2D_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCODetectEvaluator, data_root=data_root, split=test_split
            ),
            metrics_to_eval=["Det"],
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_COCO_BBOX_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.checkpoint_period = config.check_val_every_n_epoch
    pl_trainer.check_val_every_n_epoch = config.check_val_every_n_epoch
    pl_trainer.save_top_k = 1
    pl_trainer.wandb = True
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
