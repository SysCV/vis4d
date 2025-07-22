# pylint: disable=duplicate-code
"""VIT ImageNet-1k training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.engine.callbacks import EMACallback, EvaluatorCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
)
from vis4d.engine.loss_module import LossModule
from vis4d.eval.common.cls import ClassificationEvaluator
from vis4d.model.adapter import ModelEMAAdapter
from vis4d.model.cls.vit import ViTClassifer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.data_connectors.cls import (
    CONN_CLS_LOSS,
    CONN_CLS_TEST,
    CONN_CLS_TRAIN,
)
from vis4d.zoo.base.datasets.imagenet import (
    CONN_IMAGENET_CLS_EVAL,
    get_imagenet_cls_cfg,
)


def get_config() -> ExperimentConfig:
    """Returns the config dict for the ImageNet Classification task.

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################

    config = get_default_cfg(exp_name="vit_tiny_16_imagenet1k")
    config.sync_batchnorm = True
    config.check_val_every_n_epoch = 1

    ## High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 256
    params.workers_per_gpu = 8
    params.num_epochs = 300
    params.lr = 1e-3
    params.weight_decay = 0.01
    params.num_classes = 1000
    params.grad_norm_clip = 1.0
    params.accumulate_grad_batches = 1
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/imagenet1k"
    train_split = "train"
    test_split = "val"
    image_size = (224, 224)

    config.data = get_imagenet_cls_cfg(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        image_size=image_size,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    config.model = class_config(
        ModelEMAAdapter,
        model=class_config(
            ViTClassifer,
            variant="vit_tiny_patch16_224",
            num_classes=params.num_classes,
            drop_rate=0.1,
            drop_path_rate=0.1,
        ),
        decay=0.99998,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    config.loss = class_config(
        LossModule,
        losses={
            "loss": class_config(nn.CrossEntropyLoss),
            "connector": class_config(
                LossConnector, key_mapping=CONN_CLS_LOSS
            ),
        },
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################

    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                AdamW, lr=params.lr, weight_decay=params.weight_decay
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(LinearLR, estart_factor=1e-3, total_iters=10),
                    end=10,
                ),
                get_lr_scheduler_cfg(
                    class_config(
                        CosineAnnealingLR,
                        T_max=params.num_epochs,
                        eta_min=1e-9,
                    ),
                    begin=10,
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector,
        key_mapping=CONN_CLS_TRAIN,
    )

    config.test_data_connector = class_config(
        DataConnector,
        key_mapping=CONN_CLS_TEST,
    )

    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################
    callbacks = get_default_callbacks_cfg()

    # EMA callback
    callbacks.append(class_config(EMACallback))

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(ClassificationEvaluator),
            metrics_to_eval=["Cls"],
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_IMAGENET_CLS_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.gradient_clip_val = params.grad_norm_clip
    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches

    config.pl_trainer = pl_trainer

    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
