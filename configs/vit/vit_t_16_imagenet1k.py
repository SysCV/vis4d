"""ViT ImageNet training example."""
from __future__ import annotations

import pytorch_lightning as pl
from torch import nn, optim

import vis4d
from vis4d.config.default.data.dataloader import default_image_dataloader
from vis4d.config.default.data.classification import (
    classification_preprocessing,
)

from vis4d.common.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    EvaluatorCallback,
)

from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.imagenet import ImageNet
from vis4d.data.transforms.autoaugment import randaug
from vis4d.data.transforms.flip import flip_image
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.engine.connectors import data_key, pred_key
from vis4d.model.classification.vit import ClassificationViT
from vis4d.eval.classify import ClassificationEvaluator
from vis4d.optim import PolyLR, LinearLRWarmup


def get_config() -> ConfigDict:
    """Returns the config dict for the ImageNet Classification task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.cli --config configs/vit/vit_imagenet.py --config.num_epochs 100 -- config.params.lr 0.001

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################

    config = ConfigDict()
    config.n_gpus = 1
    config.work_dir = "vis4d-workspace"
    config.experiment_name = "vit_t_16_imagenet1k"
    config = set_output_dir(config)

    config.dataset_root = "./data/imagenet1k"
    config.train_split = "train"
    config.test_split = "val"

    ## High level hyper parameters
    params = ConfigDict()
    params.num_epochs = 300
    params.batch_size = 256
    params.lr = 1e-3
    params.weight_decay = 0.0001
    params.augment_proba = 1.0
    params.num_classes = 1000
    params.grad_norm_clip = 1.0
    params.accumulate_grad_batches = 1
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################

    data = ConfigDict()

    # Training Datasets
    train_dataset_cfg = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.train_split,
        num_classes=params.num_classes,
    )
    aug_cfg = (
        class_config(
            "vis4d.data.transforms.base.random_apply",
            transforms=(
                class_config(
                    "vis4d.data.transforms.flip.flip_image",
                    in_keys=(K.images,),
                    out_keys=(K.images,),
                ),
            ),
            probability=0.5,
        ),
        class_config(
            "vis4d.data.transforms.autoaugment.randaug",
            magnitude=10,
            in_keys=(K.images,),
            out_keys=(K.images,),
        ),
        class_config(
            "vis4d.data.transforms.random_erasing.random_erasing",
            probability=0.5,
            mean=(123.675, 116.28, 103.53),
            in_keys=(K.images,),
            out_keys=(K.images,),
        ),
    )
    train_preprocess_cfg = classification_preprocessing(
        224,
        224,
        augment_probability=params.augment_proba,
        augmentation_transforms=aug_cfg,
    )
    data.train_dataloader = default_image_dataloader(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        num_samples_per_gpu=params.batch_size,
        num_workers_per_gpu=9,
        shuffle=True,
    )

    # Testing Datasets
    test_dataset_cfg = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.test_split,
        num_classes=params.num_classes,
    )
    test_preprocess_cfg = classification_preprocessing(
        224, 224, augment_probability=0
    )
    test_dataloader_test = default_image_dataloader(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        num_samples_per_gpu=params.batch_size,
        num_workers_per_gpu=5,
        shuffle=False,
    )
    data.test_dataloader = {"imagenet_eval": test_dataloader_test}

    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    config.model = class_config(
        ClassificationViT,
        style="torchvision",
        vit_name="vit_t_16",
        num_classes=params.num_classes,
        image_size=224,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    config.loss = class_config(nn.CrossEntropyLoss)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################

    config.optimizers = [
        optimizer_cfg(
            optimizer=class_config(
                optim.AdamW, lr=params.lr, weight_decay=params.weight_decay
            ),
            lr_scheduler=class_config(
                optim.lr_scheduler.CosineAnnealingLR,
                T_max=params.num_epochs * 1281167 / (params.batch_size * 4)
                - 10000,
                eta_min=1e-9,
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=1e-4, warmup_steps=10000
            ),
            epoch_based_lr=False,
            epoch_based_warmup=False,
        )
    ]

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################

    # Here we define the evaluator. We use the default COCO evaluator for
    # bounding box detection. Note, that we need to define the connections
    # between the evaluator and the data connector in the data connector
    # section. And use the same name here.

    eval_callbacks = {
        "imagenet_eval": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                ClassificationEvaluator,
                num_classes=params.num_classes,
            ),
            num_epochs=params.num_epochs,
        )
    }

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################

    config.data_connector = class_config(
        StaticDataConnector,
        connections=DataConnectionInfo(
            train={"images": "images"},
            test={"images": "images"},
            loss={
                "input": pred_key("logits"),
                "target": data_key("categories"),
            },
            callbacks={
                "imagenet_eval_test": {
                    "predictions": pred_key("probs"),
                    "labels": data_key("categories"),
                }
            },
        ),
    )

    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################

    config.shared_callbacks = {
        "logging": class_config(LoggingCallback, refresh_rate=50)
    }
    config.train_callbacks = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.output_dir,
            run_every_nth_epoch=5,
            num_epochs=params.num_epochs,
        )
    }
    config.test_callbacks = {**eval_callbacks}

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    pl_trainer = ConfigDict()
    pl_trainer.wandb = True
    pl_trainer.gradient_clip_val = params.grad_norm_clip
    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches
    pl_trainer.check_val_every_n_epoch = 5

    config.pl_trainer = pl_trainer

    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
