"""FCN Segment training example."""
from __future__ import annotations

import os

from torch import nn, optim

from vis4d.config.default.data.dataloader import default_image_dl
from vis4d.config.default.data.segment import segment_preprocessing
from vis4d.config.default.data_connectors.segment import (
    CONN_FCN_LOSS,
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
)
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.default.sweep.default import linear_grid_search
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.datasets.coco import COCO
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.model.segment.fcn_resnet import FCNResNet
from vis4d.op.segment.fcn import FCNLoss
from vis4d.optim import PolyLR

# This is just for demo purposes. Uses the relative path to the vis4d root.
VIS4D_ROOT = os.path.abspath(os.path.dirname(__file__) + "../../../../")
COCO_DATA_ROOT = os.path.join(VIS4D_ROOT, "tests/vis4d-test-data/coco_test")
TRAIN_SPLIT = "train"
TEST_SPLIT = "train"


def get_config() -> ConfigDict:
    """Returns the config dict for the coco detection task.

    This is a simple example that shows how to set up a training experiment
    for the COCO detection task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/faster_rcnn_coco.py --config.num_epochs 100 -- config.params.lr 0.001

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
    config.experiment_name = "fcn_coco"
    config.save_prefix = "vis4d-workspace/test/" + config.get_ref(
        "experiment_name"
    )

    config.dataset_root = COCO_DATA_ROOT
    config.train_split = TRAIN_SPLIT
    config.test_split = TEST_SPLIT
    config.n_gpus = 1
    config.num_epochs = 40

    ## High level hyper parameters
    params = ConfigDict()
    params.batch_size = 8
    params.lr = 0.0001
    params.augment_proba = 0.5
    params.num_classes = 21
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################

    # Here we define the training and test datasets.
    # We use the COCO dataset and the default data augmentation
    # provided by vis4d.

    # Training Datasets
    dataset_cfg_train = class_config(
        COCO,
        data_root=config.dataset_root,
        split=config.train_split,
        num_workers_per_gpu=0,
        shuffle=True,
    )
    preproc = segment_preprocessing(520, 520, params.augment_proba)
    dataloader_train_cfg = default_image_dl(
        preproc,
        dataset_cfg_train,
        params.batch_size,
        num_workers_per_gpu=0,
        shuffle=True,
    )
    config.train_dl = dataloader_train_cfg

    # Test Datasets
    dataset_test_cfg = class_config(
        COCO,
        data_root=config.dataset_root,
        split=config.test_split,
        num_workers_per_gpu=0,
        shuffle=True,
    )
    preprocess_test_cfg = segment_preprocessing(
        520, 520, augment_probability=0
    )
    dataloader_cfg_test = default_image_dl(
        preprocess_test_cfg,
        dataset_test_cfg,
        batch_size=1,
        num_workers_per_gpu=0,
        shuffle=False,
    )
    config.test_dl = {"coco_eval": dataloader_cfg_test}

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    config.model = class_config(
        FCNResNet,
        base_model="resnet50",
        num_classes=params.num_classes,
        resize=(520, 520),
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    # Here we define the loss function. We use the default loss function
    # provided for the Faster RCNN model.
    # Note, that the loss functions consists of multiple loss terms which
    # are averaged using a weighted sum.

    config.loss = class_config(
        FCNLoss,
        feature_idx=[4, 5],
        loss_fn=class_config(nn.CrossEntropyLoss, ignore_index=255),
        weights=[0.5, 1],
    )

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
            optimizer=class_config(optim.Adam, lr=params.lr),
            lr_scheduler=class_config(
                PolyLR, total_iters=config.num_epochs, power=0.9
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
            train=CONN_MASKS_TRAIN,
            test=CONN_MASKS_TEST,
            loss=CONN_FCN_LOSS,
        ),
    )
    return config.value_mode()


def get_sweep() -> ConfigDict:
    """Returns the config dict for a grid search over learning rate.

    Returns:
        ConfigDict: The configuration that can be used to run a grid search.
            It can be passed to replicate_config to create a list of configs
            that can be used to run a grid search.
    """
    return linear_grid_search("params.lr", 0.001, 0.01, 3)