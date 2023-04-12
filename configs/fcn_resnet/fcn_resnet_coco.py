"""FCN-ResNet COCO training example."""
from __future__ import annotations

from torch import optim

from vis4d.config.default.data.dataloader import default_image_dataloader
from vis4d.config.default.data.seg import seg_preprocessing
from vis4d.config.default.data_connectors.seg import (
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_MULTI_SEG_LOSS,
)
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.default.sweep.default import linear_grid_search
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.datasets.coco import COCO
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.model.seg.fcn_resnet import FCNResNet
from vis4d.op.loss import MultiLevelSegLoss
from vis4d.optim import PolyLR


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

    config.dataset_root = "./data/COCO"
    config.train_split = "train2017"
    config.test_split = "val2017"
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
        use_pascal_voc_cats=True,
        minimum_box_area=10,
    )
    preproc = seg_preprocessing(520, 520, False, params.augment_proba)
    dataloader_train_cfg = default_image_dataloader(
        preproc,
        dataset_cfg_train,
        params.batch_size,
        num_workers_per_gpu=0,
        shuffle=True,
    )
    config.train_dl = dataloader_train_cfg

    # Test
    dataset_test_cfg = class_config(
        COCO,
        data_root=config.dataset_root,
        split=config.test_split,
        use_pascal_voc_cats=True,
    )
    preprocess_test_cfg = seg_preprocessing(
        520, 520, False, augment_probability=0
    )
    dataloader_cfg_test = default_image_dataloader(
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
        MultiLevelSegLoss, feature_idx=[4, 5], weights=[0.5, 1]
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
            train=CONN_MASKS_TRAIN,
            test=CONN_MASKS_TEST,
            loss=CONN_MULTI_SEG_LOSS,
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
