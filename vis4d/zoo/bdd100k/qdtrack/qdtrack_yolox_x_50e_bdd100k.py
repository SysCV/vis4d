# pylint: disable=duplicate-code
"""QDTrack-YOLOX BDD100K."""
from __future__ import annotations

import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from vis4d.config import class_config
from vis4d.config.common.datasets.bdd100k import CONN_BDD100K_TRACK_EVAL
from vis4d.config.common.models.qdtrack import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_YOLOX_TRAIN,
    get_qdtrack_yolox_cfg,
)
from vis4d.config.common.models.yolox import (
    get_yolox_callbacks_cfg,
    get_yolox_optimizers_cfg,
)
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.default.data_connectors import CONN_BBOX_2D_TRACK_VIS
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.datasets.bdd100k import bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.zoo.bdd100k.qdtrack.data_yolox import get_bdd100k_track_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="qdtrack_yolox_x_50e_bdd100k")
    config.checkpoint_period = 5
    config.check_val_every_n_epoch = 5

    # ckpt_path = (
    #     "vis4d-workspace/QDTrack/pretrained/qdtrack-yolox-ema_bdd100k.ckpt"
    # )

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 5
    params.workers_per_gpu = 4
    params.lr = 0.000625
    params.num_epochs = 50
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
    num_last_epochs, warmup_epochs = 10, 1
    config.optimizers = get_yolox_optimizers_cfg(
        params.lr, params.num_epochs, warmup_epochs, num_last_epochs
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_YOLOX_TRAIN
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
    callbacks += get_yolox_callbacks_cfg(
        switch_epoch=params.num_epochs - num_last_epochs, num_sizes=0
    )

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer, vis_freq=500),
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
