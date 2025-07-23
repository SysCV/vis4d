# pylint: disable=duplicate-code
"""CC-3DT with Faster-RCNN ResNet-50 detector using KF3D motion model."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from torch.optim.sgd import SGD

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.datasets.nuscenes import nuscenes_class_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    MultiSensorDataConnector,
)
from vis4d.eval.nuscenes import (
    NuScenesDet3DEvaluator,
    NuScenesTrack3DEvaluator,
)
from vis4d.op.base import ResNet
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.cc_3dt.data import (
    CONN_NUSC_BBOX_3D_TEST,
    CONN_NUSC_DET3D_EVAL,
    CONN_NUSC_TRACK3D_EVAL,
    get_nusc_cfg,
)
from vis4d.zoo.cc_3dt.model import CONN_BBOX_3D_TRAIN, get_cc_3dt_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="cc_3dt_frcnn_r50_fpn_kf3d_12e_nusc")

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
    version = "v1.0-trainval"
    train_split = "train"
    test_split = "val"

    data_backend = class_config(HDF5Backend)

    config.data = get_nusc_cfg(
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
        num_classes=len(nuscenes_class_map), basemodel=basemodel, fps=2
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
            param_groups=[
                {
                    "custom_keys": [
                        "faster_rcnn_head.rpn_head.rpn_cls.weight",
                        "faster_rcnn_head.rpn_head.rpn_box.weight",
                        "faster_rcnn_head.roi_head.fc_cls.weight",
                        "faster_rcnn_head.roi_head.fc_reg.weight",
                        "bbox_3d_head.dep_convs.0.weight",
                        "bbox_3d_head.dep_convs.1.weight",
                        "bbox_3d_head.dep_convs.2.weight",
                        "bbox_3d_head.dep_convs.3.weight",
                        "bbox_3d_head.dim_convs.0.weight",
                        "bbox_3d_head.dim_convs.1.weight",
                        "bbox_3d_head.dim_convs.2.weight",
                        "bbox_3d_head.dim_convs.3.weight",
                        "bbox_3d_head.rot_convs.0.weight"
                        "bbox_3d_head.rot_convs.1.weight",
                        "bbox_3d_head.rot_convs.2.weight",
                        "bbox_3d_head.rot_convs.3.weight",
                        "bbox_3d_head.cen_2d_convs.0.weight",
                        "bbox_3d_head.cen_2d_convs.1.weight",
                        "bbox_3d_head.cen_2d_convs.2.weight",
                        "bbox_3d_head.cen_2d_convs.3.weight",
                        "bbox_3d_head.fc_dep.weight",
                        "bbox_3d_head.fc_dep_uncer.weight",
                        "bbox_3d_head.fc_dim.weight",
                        "bbox_3d_head.fc_rot.weight",
                        "bbox_3d_head.fc_cen_2d.weight",
                    ],
                    "lr_mult": 10.0,
                }
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
        MultiSensorDataConnector, key_mapping=CONN_NUSC_BBOX_3D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg()

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                NuScenesDet3DEvaluator,
                data_root=data_root,
                version=version,
                split=test_split,
            ),
            save_predictions=True,
            output_dir=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_DET3D_EVAL
            ),
        )
    )

    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(NuScenesTrack3DEvaluator),
            save_predictions=True,
            output_dir=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_TRACK3D_EVAL
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
