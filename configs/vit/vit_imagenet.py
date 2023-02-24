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

    # Here we define the general config for the experiment.
    # This includes the experiment name, the dataset root, the splits
    # and the high level hyper parameters.

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

    # Here we define the training and test datasets.
    # We use the COCO dataset and the default data augmentation
    # provided by vis4d.

    # Training Datasets
    dataset_cfg_train = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.train_split,
        num_classes=params.num_classes,
        use_sample_lists=False,
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

    # Test
    dataset_cfg_test = class_config(
        ImageNet,
        data_root=config.dataset_root,
        split=config.train_split,
        num_classes=params.num_classes,
        use_sample_lists=False,
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

    # Here we define the loss function. We use the default loss function
    # provided for the Faster RCNN model.
    # Note, that the loss functions consists of multiple loss terms which
    # are averaged using a weighted sum.

    config.loss = class_config(nn.CrossEntropyLoss)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################

    # Here we define which optimizer to use. We use the default optimizer
    # provided by vis4d. By default, it consists of a optimizer, a learning
    # rate scheduler and a learning rate warmup and passes all the parameters
    # to the optimizer.
    # If required, we can also define multiple, custom optimizers and pass
    # them to the config. In order to only subscribe to a subset of the
    # parameters,
    #
    # We could add a filtering function as follows:
    # def only_encoder_params(params: Iterable[torch.Tensor], fun: Callable):
    #     return fun([p for p in params if "encoder" in p.name])
    #
    # config.optimizers = [
    #    optimizer_cfg(
    #        optimizer=class_config(only_encoder_params,
    #           fun=class_config(optim.SGD, lr=params.lr"))
    #        )
    #    )
    # ]

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

    # This defines how the output of each component is connected to the next
    # component. This is a very important part of the config. It defines the
    # data flow of the pipeline.
    # We use the default connections provided for faster_rcnn.

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
