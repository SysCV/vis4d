"""ViT ImageNet training example."""
from __future__ import annotations

from torch import nn, optim

import vis4d
from vis4d.config.default.data.dataloader import default_image_dl
from vis4d.config.default.data.classification import (
    classification_preprocessing,
)

from vis4d.common.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    EvaluatorCallback,
)

from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.datasets.imagenet import ImageNet
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.engine.connectors import data_key, pred_key
from vis4d.model.classification.vit import ClassificationViTMAE
from vis4d.eval import ClassificationEvaluator
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
    config.experiment_name = "vit_mae_imagenet"
    config.save_prefix = "vis4d-workspace/" + config.get_ref("experiment_name")

    config.dataset_root = "./data/ImageNet"
    config.train_split = "train"
    config.test_split = "val"
    config.n_gpus = 1
    config.num_epochs = 50

    ## High level hyper parameters
    params = ConfigDict()
    params.batch_size = 40
    params.lr = 0.0003125
    params.augment_proba = 0.5
    params.num_classes = 1000
    params.grad_norm_clip = 1.0
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################

    # Training Datasets
    dataset_cfg_train = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.train_split,
        num_classes=params.num_classes,
    )
    preproc = classification_preprocessing(224, 224, params.augment_proba)
    dataloader_train_cfg = default_image_dl(
        preproc,
        dataset_cfg_train,
        params.batch_size,
        num_workers_per_gpu=3,
        shuffle=True,
    )
    config.train_dl = dataloader_train_cfg

    # Testing Datasets
    dataset_cfg_test = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.test_split,
        num_classes=params.num_classes,
    )
    preproc_test = classification_preprocessing(224, 224, 0)
    dataloader_cfg_test = default_image_dl(
        preproc_test,
        dataset_cfg_test,
        batch_size=params.batch_size,
        num_workers_per_gpu=3,
        shuffle=False,
    )
    config.test_dl = {"imagenet_eval": dataloader_cfg_test}

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    config.model = class_config(
        ClassificationViTMAE,
        num_classes=params.num_classes,
        weights="./vis4d-workspace/weights/mae_pretrain_vit_base.pth",
        img_size=224,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    config.loss = class_config(nn.CrossEntropyLoss, label_smoothing=0.1)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################

    config.optimizers = [
        optimizer_cfg(
            optimizer=class_config(
                optim.AdamW, lr=params.lr, weight_decay=0.05
            ),
            lr_scheduler=class_config(
                PolyLR, max_steps=config.num_epochs, power=0.9
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=0.01, warmup_steps=5
            ),
            epoch_based=True,
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
            run_every_nth_epoch=5,
            num_epochs=config.num_epochs,
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

    config.train_callbacks = {
        "logging": class_config(LoggingCallback, refresh_rate=50),
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.save_prefix,
            run_every_nth_epoch=5,
            num_epochs=config.num_epochs,
        ),
    }
    config.test_callbacks = {**eval_callbacks}

    return config.value_mode()
