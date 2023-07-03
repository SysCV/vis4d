"""CC-3DT nuScenes inference example."""
from __future__ import annotations

import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR

from vis4d.config import class_config
from vis4d.config.common.datasets.nuscenes import get_nusc_track_cfg
from vis4d.config.common.models import get_cc_3dt_cfg
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.config.util import get_lr_scheduler_cfg, get_optimizer_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_detection_range
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    DataConnector,
    MultiSensorCallbackConnector,
    MultiSensorDataConnector,
    data_key,
    pred_key,
)
from vis4d.eval.nuscenes import NuScenesEvaluator
from vis4d.op.base import ResNet

CONN_BBOX_3D_TRAIN = {
    "images": K.images,
    "images_hw": K.input_hw,
    "intrinsics": K.intrinsics,
    "boxes2d": K.boxes2d,
    "boxes3d": K.boxes3d,
    "boxes3d_classes": K.boxes3d_classes,
    "boxes3d_track_ids": K.boxes3d_track_ids,
    "keyframes": "keyframes",
}

CONN_BBOX_3D_TEST = {
    "images": K.images,
    "images_hw": K.original_hw,
    "intrinsics": K.intrinsics,
    "extrinsics": K.extrinsics,
    "frame_ids": K.frame_ids,
}

CONN_NUSC_EVAL = {
    "tokens": data_key("token"),
    "boxes_3d": pred_key("boxes_3d"),
    "class_ids": pred_key("class_ids"),
    "scores_3d": pred_key("scores_3d"),
    "track_ids": pred_key("track_ids"),
}


def get_config() -> ExperimentConfig:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="cc_3dt_r50_kf3d")

    ckpt_path = "https://dl.cv.ethz.ch/vis4d/cc_3dt_R_50_FPN_nuscenes.pt"

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 4
    params.workers_per_gpu = 4
    params.lr = 0.01
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/nuscenes"
    # version = "v1.0-mini"
    # train_split = "mini_train"
    # test_split = "mini_val"
    version = "v1.0-trainval"
    train_split = "train"
    test_split = "val"

    data_backend = class_config(HDF5Backend)

    config.data = get_nusc_track_cfg(
        data_root=data_root,
        version=version,
        train_split=train_split,
        test_split=test_split,
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

    config.model, config.loss = get_cc_3dt_cfg(
        num_classes=10,
        basemodel=basemodel,
        detection_range=nuscenes_detection_range,
        # weights=ckpt_path,
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
                    class_config(LinearLR, start_factor=0.1, total_iters=1000),
                    end=1000,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_3D_TRAIN
    )

    config.test_data_connector = class_config(
        MultiSensorDataConnector,
        key_mapping=CONN_BBOX_3D_TEST,
        sensors=NuScenes.CAMERAS,
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                NuScenesEvaluator,
                data_root=data_root,
                version=version,
                split=test_split,
                output_dir=config.output_dir,
            ),
            save_predictions=True,
            save_prefix=config.output_dir,
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_EVAL,
                sensors=NuScenes.CAMERAS,
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
    pl_trainer.gradient_clip_val = 10
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
