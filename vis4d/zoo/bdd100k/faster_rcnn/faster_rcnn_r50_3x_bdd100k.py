"""Faster RCNN BDD100K training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.config import FieldConfigDict, class_config
from vis4d.config.common.datasets.bdd100k import (
    CONN_BDD100K_DET_EVAL,
    get_bdd100k_detection_config,
)
from vis4d.config.common.models import get_faster_rcnn_cfg
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.default.data_connectors import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_BBOX_2D_VIS,
)
from vis4d.config.util import get_optimizer_cfg
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.engine.optim.warmup import LinearLRWarmup
from vis4d.eval.bdd100k import BDD100KDetectEvaluator
from vis4d.op.base import ResNet
from vis4d.vis.image import BoundingBoxVisualizer


def get_config() -> FieldConfigDict:
    """Returns the Faster-RCNN config dict for the BDD100K detection task.

    Returns:
        FieldConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="faster_rcnn_r50_3x_bdd100k")

    # High level hyper parameters
    params = FieldConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.02
    params.num_epochs = 36
    params.num_classes = 10
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/bdd100k/images/100k"
    train_split = "train"
    test_split = "val"

    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_detection_config(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        multi_scale=True,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet50", pretrained=True, trainable_layers=3
    )

    config.model, config.loss = get_faster_rcnn_cfg(
        num_classes=params.num_classes, basemodel=basemodel
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_scheduler=class_config(
                MultiStepLR, milestones=[24, 33], gamma=0.1
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=0.001, warmup_steps=500
            ),
            epoch_based_lr=True,
            epoch_based_warmup=False,
        )
    ]

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
    callbacks = get_default_callbacks_cfg(config)

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer, vis_freq=100),
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
                BDD100KDetectEvaluator,
                annotation_path="data/bdd100k/labels/det_20/det_val.json",
                config_path="det",
            ),
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_DET_EVAL
            ),
            metrics_to_eval=[BDD100KDetectEvaluator.METRICS_DET],
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
