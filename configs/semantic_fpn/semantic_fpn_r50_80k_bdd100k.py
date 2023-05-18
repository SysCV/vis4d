"""Semantic FPN BDD100K training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch import optim

from vis4d.config.base.datasets.bdd100k.semantic_segmentation import (
    CONN_BDD100K_SEG_EVAL,
    get_bdd100k_segmentation_config,
)
from vis4d.config.default import get_pl_trainer_config
from vis4d.config.default.data_connectors.seg import (
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_SEG_LOSS,
    CONN_SEG_VIS,
)
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    LoggingCallback,
    VisualizerCallback,
)
from vis4d.engine.connectors import DataConnector
from vis4d.engine.optim import PolyLR
from vis4d.engine.optim.warmup import LinearLRWarmup
from vis4d.eval.seg.bdd100k import BDD100KSegEvaluator
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss
from vis4d.vis.image import SegMaskVisualizer


def get_config() -> ConfigDict:
    """Returns the config dict for the BDD100K semantic segmentation task.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()

    config.work_dir = "vis4d-workspace"
    config.experiment_name = "semantic_fpn_r50_80k_bdd100k"
    config = set_output_dir(config)
    config.sync_batchnorm = True
    config.val_check_interval = 4000

    ## High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_steps = 80000
    params.num_classes = 19
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/bdd100k/images/10k"
    train_split = "train"
    test_split = "val"

    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_segmentation_config(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
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
    callbacks = []

    # Logger
    callbacks.append(
        class_config(LoggingCallback, epoch_based=False, refresh_rate=50)
    )

    # # Checkpoint
    # callbacks.append(
    #     class_config(CheckpointCallback, save_prefix=config.output_dir)
    # )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KSegEvaluator,
                annotation_path="data/bdd100k/labels/sem_seg_val_rle.json",
            ),
            test_connector=CONN_BDD100K_SEG_EVAL,
        )
    )

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(SegMaskVisualizer, vis_freq=20),
            save_prefix=config.output_dir,
            test_connector=CONN_SEG_VIS,
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_pl_trainer_config(config)
    pl_trainer.epoch_based = False
    pl_trainer.max_steps = params.num_steps

    val_freq = 4000
    pl_trainer.checkpoint_period = val_freq
    pl_trainer.val_check_interval = val_freq
    pl_trainer.check_val_every_n_epoch = None

    pl_trainer.sync_batchnorm = True
    # pl_trainer.precision = 16

    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
