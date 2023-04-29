"""Semantic FPN BDD100K training example."""
from __future__ import annotations

from torch import optim

from vis4d.common.callbacks import EvaluatorCallback
from vis4d.config.base.datasets.bdd100k_segmentation import (
    CONN_BDD100K_SEG_EVAL,
    get_bdd100k_segmentation_config,
)
from vis4d.config.default.data_connectors.seg import (
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_SEG_LOSS,
)
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.default.runtime import (
    get_generic_callback_config,
    set_output_dir,
)
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.eval.seg.bdd100k import BDD100KSegEvaluator
from vis4d.model.seg.semantic_fpn import SemanticFPN
from vis4d.op.loss import SegCrossEntropyLoss
from vis4d.optim import PolyLR
from vis4d.optim.warmup import LinearLRWarmup


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
    config.experiment_name = "test/semantic_fpn_bdd100k"
    config = set_output_dir(config)

    ## High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_steps = 40000
    params.num_epochs = 45
    params.augment_prob = 0.5
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
        StaticDataConnector,
        connections=DataConnectionInfo(
            train=CONN_MASKS_TRAIN,
            test=CONN_MASKS_TEST,
            loss=CONN_SEG_LOSS,
            callbacks={"bdd100k_eval": CONN_BDD100K_SEG_EVAL},
        ),
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################

    eval_callbacks = {
        "bdd100k_eval": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KSegEvaluator,
                annotation_path="data/bdd100k/labels/sem_seg_val_rle.json",
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Generic callbacks
    logger_callback, ckpt_callback = get_generic_callback_config(
        config, params
    )

    # Assign the defined callbacks to the config
    config.shared_callbacks = {**logger_callback}

    config.train_callbacks = {**ckpt_callback}
    config.test_callbacks = {**eval_callbacks}

    return config.value_mode()
