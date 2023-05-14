"""Semantic FPN BDD100K training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch import optim

from vis4d.config.base.datasets.shift.tasks import (
    get_shift_segmentation_config,
)
from vis4d.config.default import (
    get_callbacks_config,
    get_pl_trainer_config,
    set_output_dir,
)
from vis4d.config.default.data_connectors.seg import (
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_SEG_EVAL,
    CONN_SEG_LOSS,
)
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import DataConnector
from vis4d.engine.optim import PolyLR
from vis4d.engine.optim.warmup import LinearLRWarmup
from vis4d.eval.shift.seg import SHIFTSegEvaluator
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss


def get_config() -> ConfigDict:
    """Returns the config dict for the SHIFT semantic segmentation task.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()

    config.work_dir = "vis4d-workspace"
    config.experiment_name = "shift_semantic_fpn"
    config = set_output_dir(config)

    ## High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_steps = 40000
    params.num_epochs = 45
    params.augment_prob = 0.5
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

    config.data = get_shift_segmentation_config(
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
    config.loss = class_config(SegCrossEntropyLoss)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_config(
            optimizer=class_config(
                optim.SGD, lr=params.lr, momentum=0.9, weight_decay=0.0005
            ),
            lr_scheduler=class_config(
                PolyLR, max_steps=params.num_steps, min_lr=0.0001, power=0.9
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=0.001, warmup_steps=500
            ),
            epoch_based_lr=False,
            epoch_based_warmup=False,
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.data_connector = class_config(
        DataConnector,
        train=CONN_MASKS_TRAIN,
        test=CONN_MASKS_TEST,
        loss=CONN_SEG_LOSS,
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_callbacks_config(config)

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                SHIFTSegEvaluator,
            ),
            test_connector=CONN_SEG_EVAL,
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_pl_trainer_config(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.wandb = True
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
