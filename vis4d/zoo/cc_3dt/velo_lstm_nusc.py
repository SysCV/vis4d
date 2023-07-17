# pylint: disable=duplicate-code
"""CC-3DT VeloLSTM on nuScenes."""
from __future__ import annotations

import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.config import class_config
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.typing import (
    DataConfig,
    ExperimentConfig,
    ExperimentParameters,
)
from vis4d.config.util import (
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
    get_train_dataloader_cfg,
)

from vis4d.engine.connectors import (
    DataConnector,
    LossConnector,
    data_key,
    pred_key,
)
from vis4d.engine.loss_module import LossModule

from vis4d.data.datasets.trajectory import Trajectory
from vis4d.model.motion.velo_lstm import VeloLSTM, VeloLSTMLoss

TRAJ_TRAIN = {"pred_traj": "pred_traj"}
TRAJ_LOSS = {
    "loc_preds": pred_key("loc_preds"),
    "loc_refines": pred_key("loc_refines"),
    "gt_traj": data_key("gt_traj"),
}


def get_config() -> ExperimentConfig:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="velo_lstm_nusc")

    config.seed = 100

    # Hyper Parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 32
    params.workers_per_gpu = 4
    params.lr = 0.005
    params.num_epochs = 100
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = DataConfig()

    train_dataset_cfg = class_config(
        Trajectory,
        method_name="cc_3dt_frcnn_r101_fpn",
        data_root="data/nuscenes",
        version="v1.0-trainval",
        split="train",
        pure_detection="./vis4d-workspace/pure_det/cc_3dt_frcnn_r101_fpn.json",
    )

    data.train_dataloader = get_train_dataloader_cfg(
        dataset_cfg=train_dataset_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        collate_keys=["gt_traj", "pred_traj"],
    )

    data.test_dataloader = None

    config.data = data

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model = class_config(VeloLSTM)

    config.loss = class_config(
        LossModule,
        losses=[
            {
                "loss": class_config(VeloLSTMLoss),
                "connector": class_config(
                    LossConnector, key_mapping=TRAJ_LOSS
                ),
            }
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                Adam, lr=params.lr, amsgrad=True, weight_decay=0.0001
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(
                        MultiStepLR, milestones=[20, 40, 60, 80], gamma=0.5
                    ),
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=TRAJ_TRAIN
    )

    config.test_data_connector = None

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir, refresh_rate=10)

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.gradient_clip_val = 3
    pl_trainer.checkpoint_period = 20
    pl_trainer.check_val_every_n_epoch = 101
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
