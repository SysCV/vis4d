# pylint: disable=duplicate-code
"""BEVFormer tiny with ResNet-50 backbone."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import CallbackConnector, MultiSensorDataConnector
from vis4d.eval.nuscenes import NuScenesDet3DEvaluator
from vis4d.model.detect3d.bevformer import BEVFormer
from vis4d.op.base import ResNet
from vis4d.op.detect3d.bevformer import BEVFormerHead
from vis4d.op.detect3d.bevformer.encoder import (
    BEVFormerEncoder,
    BEVFormerEncoderLayer,
)
from vis4d.op.detect3d.bevformer.spatial_cross_attention import (
    MSDeformableAttention3D,
    SpatialCrossAttention,
)
from vis4d.op.detect3d.bevformer.transformer import PerceptionTransformer
from vis4d.op.fpp.fpn import FPN
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.bevformer.data import (
    CONN_NUSC_BBOX_3D_TEST,
    CONN_NUSC_DET3D_EVAL,
    get_nusc_cfg,
    nuscenes_class_map,
)


def get_config() -> ExperimentConfig:
    """Returns the config dict for BEVFormer on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="bevformer_tiny")

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 1
    params.workers_per_gpu = 4
    params.lr = 2e-4
    params.num_epochs = 24
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
        scale_factor=0.5,
        style="pytorch",
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        ResNet, resnet_name="resnet50", trainable_layers=3, pretrained=True
    )

    config.model = class_config(
        BEVFormer,
        basemodel=basemodel,
        fpn=class_config(
            FPN,
            in_channels_list=[2048],
            out_channels=256,
            extra_blocks=None,
            start_index=5,
        ),
        pts_bbox_head=class_config(
            BEVFormerHead,
            transformer=class_config(
                PerceptionTransformer,
                encoder=class_config(
                    BEVFormerEncoder,
                    layer=class_config(
                        BEVFormerEncoderLayer,
                        cross_attn=class_config(
                            SpatialCrossAttention,
                            deformable_attention=class_config(
                                MSDeformableAttention3D,
                                num_levels=1,
                            ),
                        ),
                    ),
                    num_layers=3,
                ),
            ),
            bev_h=50,
            bev_w=50,
        ),
        weights="https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth",  # pylint: disable=line-too-long
    )

    config.loss = None

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(AdamW, lr=params.lr, weight_decay=0.01),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(
                        LinearLR, start_factor=1.0 / 3, total_iters=500
                    ),
                    end=500,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(CosineAnnealingLR, T_max=params.num_epochs),
                ),
            ],
            param_groups=[{"custom_keys": ["basemodel"], "lr_mult": 0.1}],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = None

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
                class_map=nuscenes_class_map,
                velocity_thres=0.2,
            ),
            save_predictions=True,
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_NUSC_DET3D_EVAL
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
    pl_trainer.gradient_clip_val = 35
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
