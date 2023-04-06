"""Faster RCNN COCO training example."""
from __future__ import annotations

import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.op.base import ResNet

from vis4d.common.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.config.base.datasets.coco_detection import (
    CONN_COCO_BBOX_EVAL,
    CONN_BBOX_2D_VIS,
    get_coco_detection_config,
)
from vis4d.config.base.models.faster_rcnn import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_ROI_LOSS_2D,
    CONN_RPN_LOSS_2D,
    get_model_cfg,
)

from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.default.runtime import (
    set_output_dir,
    get_generic_callback_config,
    get_pl_trainer_args,
)

from vis4d.config.util import ConfigDict, class_config

from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.eval.detect.coco import COCOEvaluator

from vis4d.optim.warmup import LinearLRWarmup
from vis4d.vis.image import BoundingBoxVisualizer


def get_config() -> ConfigDict:
    """Returns the config dict for the coco detection task.

    This is a simple example that shows how to set up a training experiment
    for the COCO detection task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.cli fit --config configs/faster_rcnn/faster_rcnn_coco.py --config.params.lr 0.001

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()

    config.work_dir = "vis4d-workspace"
    config.experiment_name = "faster_rcnn_r50_fpn_coco"
    config = set_output_dir(config)

    # High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.lr = 0.02
    params.num_epochs = 12
    params.augment_proba = 0.5
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/coco"
    train_split = "train2017"
    test_split = "val2017"

    data_backend = class_config(HDF5Backend)

    config.data = get_coco_detection_config(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        samples_per_gpu=params.samples_per_gpu,
        data_backend=data_backend,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    backbone = class_config(
        ResNet, resnet_name="resnet50", pretrained=True, trainable_layers=3
    )

    config.model, config.loss = get_model_cfg(
        num_classes=params.num_classes, backbone=backbone
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_config(
            optimizer=class_config(
                optim.SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_scheduler=class_config(
                MultiStepLR, milestones=[8, 11], gamma=0.1
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
    config.data_connector = class_config(
        StaticDataConnector,
        connections=DataConnectionInfo(
            train=CONN_BBOX_2D_TRAIN,
            test=CONN_BBOX_2D_TEST,
            loss={**CONN_RPN_LOSS_2D, **CONN_ROI_LOSS_2D},
            callbacks={
                "coco_eval_test": CONN_COCO_BBOX_EVAL,
                "bbox_vis_test": CONN_BBOX_2D_VIS,
            },
        ),
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################
    eval_callbacks = {
        "coco_eval": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCOEvaluator,
                data_root=data_root,
                split=test_split,
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    ######################################################
    ##                    VISUALIZER                    ##
    ######################################################
    vis_callbacks = {
        "bbox_vis": class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer),
            save_prefix=config.output_dir,
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
    config.shared_callbacks = {
        **logger_callback,
        **eval_callbacks,
    }

    config.train_callbacks = {
        **ckpt_callback,
    }
    config.test_callbacks = {
        **vis_callbacks,
    }

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_pl_trainer_args()
    pl_trainer.max_epochs = params.num_epochs
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
