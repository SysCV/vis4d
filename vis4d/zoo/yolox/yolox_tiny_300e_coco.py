# pylint: disable=duplicate-code
"""YOLOX COCO."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim import SGD

from vis4d.config import FieldConfigDict, class_config
from vis4d.config.common.datasets.coco.yolox import (
    CONN_COCO_BBOX_EVAL,
    get_coco_yolox_cfg,
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
from vis4d.config.util import get_optimizer_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import (
    EvaluatorCallback,
    VisualizerCallback,
    YOLOXModeSwitchCallback,
)
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
    data_key,
    pred_key,
)
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim.scheduler import YOLOXCosineAnnealingLR
from vis4d.engine.optim.warmup import QuadraticLRWarmup
from vis4d.eval.coco import COCODetectEvaluator
from vis4d.model.detect.yolox import YOLOX
from vis4d.op.base import CSPDarknet
from vis4d.op.detect.yolox import YOLOXHead, YOLOXHeadLoss
from vis4d.op.fpp import YOLOXPAFPN
from vis4d.vis.image import BoundingBoxVisualizer

CONN_BBOX_2D_TRAIN = {"images": K.images}

CONN_YOLOX_LOSS_2D = {
    "cls_outs": pred_key("cls_score"),
    "reg_outs": pred_key("bbox_pred"),
    "obj_outs": pred_key("objectness"),
    "target_boxes": data_key(K.boxes2d),
    "target_class_ids": data_key(K.boxes2d_classes),
    "images_hw": data_key(K.input_hw),
}


def get_config() -> FieldConfigDict:
    """Returns the YOLOX config dict for the coco detection task.

    Returns:
        FieldConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="yolox_tiny_300e_coco")
    config.check_val_every_n_epoch = 10

    # High level hyper parameters
    params = FieldConfigDict()
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
        test_image_size=(416, 416),
        resize_size=(320, 320),
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        CSPDarknet, deepen_factor=0.33, widen_factor=0.375
    )
    fpn = class_config(
        YOLOXPAFPN,
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
    )
    yolox_head = class_config(
        YOLOXHead,
        num_classes=params.num_classes,
        in_channels=96,
        feat_channels=96,
    )
    config.model = class_config(
        YOLOX,
        num_classes=params.num_classes,
        basemodel=basemodel,
        fpn=fpn,
        yolox_head=yolox_head,
    )

    config.loss = class_config(
        LossModule,
        losses=[
            {
                "loss": class_config(
                    YOLOXHeadLoss, num_classes=params.num_classes
                ),
                "connector": class_config(
                    LossConnector, key_mapping=CONN_YOLOX_LOSS_2D
                ),
            },
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    steps_per_epoch, num_last_epochs, warmup_epochs = 1833, 15, 5
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD,
                lr=params.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True,
            ),
            lr_scheduler=class_config(
                YOLOXCosineAnnealingLR,
                max_steps=(params.num_epochs - num_last_epochs)
                * steps_per_epoch,
                eta_min=params.lr * 0.05,
            ),
            lr_warmup=class_config(
                QuadraticLRWarmup,
                warmup_ratio=1.0,
                warmup_steps=steps_per_epoch * warmup_epochs,
            ),
            epoch_based_lr=False,
            epoch_based_warmup=False,
            param_groups_cfg=[
                {
                    "custom_keys": ["basemodel", "fpn", "yolox_head"],
                    "norm_decay_mult": 0.0,
                },
                {
                    "custom_keys": ["basemodel", "fpn", "yolox_head"],
                    "bias_decay_mult": 0.0,
                },
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
    callbacks = get_default_callbacks_cfg(config)

    # YOLOX mode switch
    callbacks.append(
        class_config(
            YOLOXModeSwitchCallback,
            switch_epoch=params.num_epochs - num_last_epochs,
        )
    )

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
    pl_trainer.wandb = True
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
