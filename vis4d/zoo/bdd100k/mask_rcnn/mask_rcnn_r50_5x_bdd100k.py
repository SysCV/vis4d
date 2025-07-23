# pylint: disable=duplicate-code
"""Mask RCNN BDD100K training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from torch.optim.sgd import SGD

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.const import CommonKeys as K
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector, DataConnector
from vis4d.eval.bdd100k import BDD100KDetectEvaluator
from vis4d.op.base import ResNet
from vis4d.vis.image import SegMaskVisualizer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.data_connectors import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_INS_MASK_2D_VIS,
)
from vis4d.zoo.base.datasets.bdd100k import (
    CONN_BDD100K_INS_EVAL,
    get_bdd100k_detection_config,
)
from vis4d.zoo.base.models.mask_rcnn import get_mask_rcnn_cfg


def get_config() -> ExperimentConfig:
    """Returns the Mask R-CNN config dict for BDD100K instance segmentation.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="mask_rcnn_r50_5x_bdd100k")
    config.check_val_every_n_epoch = 5

    # High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.02
    params.num_epochs = 60
    params.num_classes = 8
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/bdd100k/images/10k"
    train_split = "train"
    test_split = "val"

    data_backend = class_config(HDF5Backend)

    config.data = get_bdd100k_detection_config(
        data_root=data_root,
        train_split=train_split,
        train_keys_to_load=(K.images, K.boxes2d, K.instance_masks),
        test_split=test_split,
        test_keys_to_load=(K.images, K.original_images),
        ins_seg=True,
        multi_scale=True,
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet50", pretrained=True, trainable_layers=3
    )

    config.model, config.loss = get_mask_rcnn_cfg(
        num_classes=params.num_classes, basemodel=basemodel, no_overlap=True
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
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
                    class_config(MultiStepLR, milestones=[40, 55], gamma=0.1),
                ),
            ],
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
    callbacks = get_default_callbacks_cfg()

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(SegMaskVisualizer, vis_freq=25),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_INS_MASK_2D_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KDetectEvaluator,
                annotation_path="data/bdd100k/labels/ins_seg_val_rle.json",
                config_path="ins_seg",
            ),
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_INS_EVAL
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
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
