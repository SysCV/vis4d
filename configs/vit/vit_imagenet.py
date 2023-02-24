"""ViT ImageNet training example."""
from __future__ import annotations

from torch import nn, optim

from vis4d.config.default.data.dataloader import default_image_dl
from vis4d.config.default.data.classification import (
    classification_preprocessing,
)

from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.datasets.imagenet import ImageNet
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.engine.connectors import data_key, pred_key
from vis4d.model.classification.vit import ClassificationViT
from vis4d.optim import PolyLR


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
    config.experiment_name = "vit_imagenet"
    config.save_prefix = "vis4d-workspace/test/" + config.get_ref(
        "experiment_name"
    )

    config.dataset_root = "./data/ImageNet"
    config.train_split = "train"
    config.test_split = "val"
    config.n_gpus = 1
    config.num_epochs = 40

    ## High level hyper parameters
    params = ConfigDict()
    params.batch_size = 8
    params.lr = 0.0001
    params.augment_proba = 0.5
    params.num_classes = 1000
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
        use_sample_lists=True,
    )
    preproc = classification_preprocessing(224, 224, params.augment_proba)
    dataloader_train_cfg = default_image_dl(
        preproc,
        dataset_cfg_train,
        params.batch_size,
        num_workers_per_gpu=0,
        shuffle=True,
    )
    config.train_dl = dataloader_train_cfg

    # Testing Datasets
    dataset_cfg_test = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.train_split,
        num_classes=params.num_classes,
        use_sample_lists=True,
    )
    preproc_test = classification_preprocessing(224, 224, 0)
    dataloader_cfg_test = default_image_dl(
        preproc_test,
        dataset_cfg_test,
        batch_size=1,
        num_workers_per_gpu=0,
        shuffle=False,
    )
    config.test_dl = {"imagenet_eval": dataloader_cfg_test}

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    config.model = class_config(
        ClassificationViT,
        vit_name="vit_b_16",
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
            optimizer=class_config(optim.AdamW, lr=params.lr),
            lr_scheduler=class_config(
                PolyLR, max_steps=config.num_epochs, power=0.9
            ),
            lr_warmup=None,
        )
    ]

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
        ),
    )
    return config.value_mode()
