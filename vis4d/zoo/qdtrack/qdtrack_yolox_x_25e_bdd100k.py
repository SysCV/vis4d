# pylint: disable=duplicate-code
"""QDTrack with YOLOX-x on BDD100K."""
from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.datasets.bdd100k import bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.zoo.base.data_connectors import CONN_BBOX_2D_TRACK_VIS
from vis4d.zoo.base.datasets.bdd100k import CONN_BDD100K_TRACK_EVAL
from vis4d.zoo.base.models.qdtrack import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    get_qdtrack_yolox_cfg,
)
from vis4d.zoo.base.models.yolox import (
    get_yolox_callbacks_cfg,
    get_yolox_optimizers_cfg,
)
from vis4d.zoo.qdtrack.data_yolox import get_bdd100k_track_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="qdtrack_yolox_x_25e_bdd100k")
    config.checkpoint_period = 5
    config.check_val_every_n_epoch = 5

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 8  # batch size = 8 GPUs * 8 samples per GPU = 64
    params.workers_per_gpu = 8
    params.lr = 0.001
    params.num_epochs = 25
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_track_cfg(
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    num_classes = len(bdd100k_track_map)
    weights = (
        "mmdet://yolox/yolox_x_8x8_300e_coco/"
        "yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"
    )
    config.model, config.loss = get_qdtrack_yolox_cfg(
        num_classes, "xlarge", weights=weights
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    # we use a schedule with 50 epochs, but only train for 25 epochs
    num_total_epochs, num_last_epochs = 50, 10
    config.optimizers = get_yolox_optimizers_cfg(
        params.lr, num_total_epochs, 1, num_last_epochs
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
        refresh_rate=config.log_every_n_steps
    )

    # YOLOX callbacks
    callbacks += get_yolox_callbacks_cfg(
        switch_epoch=num_total_epochs - num_last_epochs, num_sizes=0
    )

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBoxVisualizer, vis_freq=500, image_mode="BGR"
            ),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_2D_TRACK_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KTrackEvaluator,
                annotation_path="data/bdd100k/labels/box_track_20/val/",
            ),
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_TRACK_EVAL
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
    pl_trainer.check_val_every_n_epoch = config.check_val_every_n_epoch
    pl_trainer.checkpoint_callback = class_config(
        ModelCheckpoint,
        dirpath=config.get_ref("output_dir") + "/checkpoints",
        verbose=True,
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=config.checkpoint_period,
        save_top_k=5,
        mode="max",
        monitor="step",
    )
    pl_trainer.wandb = True
    pl_trainer.precision = "16-mixed"
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
