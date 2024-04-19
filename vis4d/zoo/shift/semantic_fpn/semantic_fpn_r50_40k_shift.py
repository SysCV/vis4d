# pylint: disable=duplicate-code
"""Semantic FPN SHIFT training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch import optim
from torch.optim.lr_scheduler import LinearLR

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
)
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim import PolyLR
from vis4d.eval.shift import SHIFTSegEvaluator
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss
from vis4d.vis.image import SegMaskVisualizer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.data_connectors.seg import (
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_SEG_EVAL,
    CONN_SEG_LOSS,
    CONN_SEG_VIS,
)
from vis4d.zoo.base.datasets.shift import get_shift_sem_seg_config


def get_config() -> ExperimentConfig:
    """Returns the config dict for the BDD100K semantic segmentation task.

    Returns:
        ExperimentParameters: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="semantic_fpn_r50_40k_shift")
    config.sync_batchnorm = True
    config.val_check_interval = 2000
    config.check_val_every_n_epoch = None

    ## High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_steps = 160000
    params.num_classes = 23
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/shift/"
    views_to_load = ["front"]
    train_split = "train"
    test_split = "val"
    domain_attr = [{"weather_coarse": "clear", "timeofday_coarse": "daytime"}]
    data_backend = class_config(HDF5Backend)

    config.data = get_shift_sem_seg_config(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        train_views_to_load=views_to_load,
        test_views_to_load=views_to_load,
        train_attributes_to_load=domain_attr,
        test_attributes_to_load=domain_attr,
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                   MODEL & LOSS                   ##
    ######################################################
    config.model = class_config(SemanticFPN, num_classes=params.num_classes)
    config.loss = class_config(
        LossModule,
        losses=[
            {
                "loss": class_config(SegCrossEntropyLoss),
                "connector": class_config(
                    LossConnector, key_mapping=CONN_SEG_LOSS
                ),
            },
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                optim.SGD, lr=params.lr, momentum=0.9, weight_decay=0.0005
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(
                        LinearLR, start_factor=0.001, total_iters=500
                    ),
                    end=500,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(
                        PolyLR,
                        max_steps=params.num_steps,
                        min_lr=0.0001,
                        power=0.9,
                    ),
                    epoch_based=False,
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_MASKS_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_MASKS_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    callbacks = get_default_callbacks_cfg(
        config.output_dir,
        epoch_based=False,
        checkpoint_period=config.val_check_interval,
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                SHIFTSegEvaluator,
            ),
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_SEG_EVAL
            ),
        )
    )

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(SegMaskVisualizer, vis_freq=20),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_SEG_VIS
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.epoch_based = False
    pl_trainer.max_steps = params.num_steps

    pl_trainer.checkpoint_period = config.val_check_interval
    pl_trainer.val_check_interval = config.val_check_interval
    pl_trainer.check_val_every_n_epoch = config.check_val_every_n_epoch

    pl_trainer.sync_batchnorm = config.sync_batchnorm
    # pl_trainer.precision = 16

    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
