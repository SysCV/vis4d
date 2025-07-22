"""FCN-ResNet COCO training example."""

from __future__ import annotations

import lightning.pytorch as pl
from torch.optim.lr_scheduler import LinearLR
from torch.optim.sgd import SGD

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import DataConnector, LossConnector
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim import PolyLR
from vis4d.model.seg.fcn_resnet import FCNResNet
from vis4d.op.loss import MultiLevelSegLoss
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.data_connectors.seg import (
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_MULTI_SEG_LOSS,
)
from vis4d.zoo.base.datasets.coco import get_coco_sem_seg_cfg


def get_config() -> ExperimentConfig:
    """Returns the config dict for the COCO semantic segmentation task.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="fcn_coco")
    config.sync_batchnorm = True
    config.val_check_interval = 2000
    config.check_val_every_n_epoch = None

    ## High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_steps = 40000
    params.num_classes = 21
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/COCO"
    train_split = "train2017"
    test_split = "val2017"
    image_size = (520, 520)

    data_backend = class_config(HDF5Backend)

    config.data = get_coco_sem_seg_cfg(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        image_size=image_size,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    config.model = class_config(
        FCNResNet,
        base_model="resnet50",
        num_classes=params.num_classes,
        resize=image_size,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    config.loss = class_config(
        LossModule,
        losses={
            "loss": class_config(
                MultiLevelSegLoss, feature_idx=[4, 5], weights=[0.5, 1]
            ),
            "connector": class_config(
                LossConnector, key_mapping=CONN_MULTI_SEG_LOSS
            ),
        },
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0005
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
                    class_config(
                        PolyLR,
                        max_steps=params.num_steps,
                        min_lr=0.0001,
                        power=0.9,
                    ),
                    epoch_based=False,
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_MASKS_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_MASKS_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    callbacks = get_default_callbacks_cfg(epoch_based=False)

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.epoch_based = False
    pl_trainer.max_steps = params.num_steps

    pl_trainer.checkpoint_period = config.val_check_interval
    pl_trainer.val_check_interval = config.val_check_interval
    pl_trainer.check_val_every_n_epoch = config.check_val_every_n_epoch

    pl_trainer.sync_batchnorm = config.sync_batchnorm
    # pl_trainer.precision = 16

    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks
    return config.value_mode()
