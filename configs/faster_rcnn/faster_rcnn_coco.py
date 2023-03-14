"""Faster RCNN COCO training example."""
from __future__ import annotations

import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    LoggingCallback,
    VisualizerCallback,
)
from vis4d.config.default.data.dataloader import default_image_dataloader
from vis4d.config.default.data.detect import det_preprocessing
from vis4d.config.default.data_connectors import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_BBOX_2D_VIS,
    CONN_COCO_BBOX_EVAL,
    CONN_ROI_LOSS_2D,
    CONN_RPN_LOSS_2D,
)
from vis4d.config.default.loss.faster_rcnn_loss import (
    get_default_faster_rcnn_loss,
)
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.default.sweep.default import linear_grid_search
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as CK
from vis4d.data.datasets.coco import COCO
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.model.detect.faster_rcnn import FasterRCNN
from vis4d.op.detect.faster_rcnn import (
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.optim.warmup import LinearLRWarmup
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.data.io.hdf5 import HDF5Backend


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
    config.n_gpus = 8
    config.work_dir = "vis4d-workspace"
    config.experiment_name = "faster_rcnn_r50_fpn_coco"
    config = set_output_dir(config)

    config.dataset_root = "data/coco"
    config.train_split = "train2017"
    config.test_split = "val2017"

    ## High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 4
    params.lr = 0.02
    params.num_epochs = 12
    params.augment_proba = 0.5
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################

    # Here we define the training and test datasets.
    # We use the COCO dataset and the default data augmentation
    # provided by vis4d.
    data = ConfigDict()
    data_backend = HDF5Backend()

    # Train
    train_dataset_cfg = class_config(
        COCO,
        keys=(CK.images, CK.boxes2d, CK.boxes2d_classes),
        data_root=config.dataset_root,
        split=config.train_split,
        data_backend=data_backend,
    )
    train_preprocess_cfg = det_preprocessing(800, 1333, params.augment_proba)
    data.train_dataloader = default_image_dataloader(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        num_samples_per_gpu=params.samples_per_gpu,
        shuffle=True,
    )

    # Test
    test_dataset_cfg = class_config(
        COCO,
        keys=(CK.images, CK.boxes2d, CK.boxes2d_classes),
        data_root=config.dataset_root,
        split=config.test_split,
        data_backend=data_backend,
    )
    test_preprocess_cfg = det_preprocessing(800, 1333, augment_probability=0)
    test_dataloader_test = default_image_dataloader(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        num_samples_per_gpu=1,
        train=False,
    )
    data.test_dataloader = {"coco_eval": test_dataloader_test}

    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    # Here we define the model. We use the default Faster RCNN model
    # provided by vis4d.
    config.gen = ConfigDict()
    config.gen.anchor_generator = class_config(get_default_anchor_generator)
    config.gen.rcnn_box_encoder = class_config(get_default_rcnn_box_encoder)
    config.gen.rpn_box_encoder = class_config(get_default_rpn_box_encoder)

    config.model = class_config(
        FasterRCNN,
        # weights="mmdet",
        num_classes=params.num_classes,
        rpn_box_encoder=config.gen.rpn_box_encoder,
        rcnn_box_encoder=config.gen.rcnn_box_encoder,
        anchor_generator=config.gen.anchor_generator,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    # Here we define the loss function. We use the default loss function
    # provided for the Faster RCNN model.
    # Note, that the loss functions consists of multiple loss terms which
    # are averaged using a weighted sum.

    config.loss = class_config(
        get_default_faster_rcnn_loss,
        rpn_box_encoder=config.gen.rpn_box_encoder,
        rcnn_box_encoder=config.gen.rcnn_box_encoder,
        anchor_generator=config.gen.anchor_generator,
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
            optimizer=class_config(
                optim.SGD, lr=params.lr, weight_decay=0.0001
            ),
            lr_scheduler=class_config(
                MultiStepLR, milestones=[8, 11], gamma=0.1
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=0.001, warmup_steps=500
            ),
            epoch_based_lr=True,
            epoch_based_warmup=False,
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
            train=CONN_BBOX_2D_TRAIN,
            test=CONN_BBOX_2D_TEST,
            loss={**CONN_RPN_LOSS_2D, **CONN_ROI_LOSS_2D},
            callbacks={
                "coco_eval_test": CONN_COCO_BBOX_EVAL,
                # "bbox_vis_test": CONN_BBOX_2D_VIS,
            },
        ),
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################

    # Here we define the evaluator. We use the default COCO evaluator for
    # bounding box detection. Note, that we need to define the connections
    # between the evaluator and the data connector in the data connector
    # section. And use the same name here.

    eval_callbacks = {
        "coco_eval": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCOEvaluator,
                data_root=config.dataset_root,
                split=config.test_split,
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    ######################################################
    ##                    VISUALIZER                    ##
    ######################################################
    # Here we define the visualizer. We use the default visualizer for
    # bounding box detection. Note, that we need to define the connections
    # between the visualizer and the data connector in the data connector
    # section. And use the same name here.

    # vis_callbacks = {
    #     "bbox_vis": class_config(
    #         VisualizerCallback,
    #         visualizer=class_config(BoundingBoxVisualizer),
    #         save_prefix=config.output_dir,
    #         run_every_nth_epoch=1,
    #         num_epochs=params.num_epochs,
    #     )
    # }

    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################
    # Here we define general, all purpose callbacks. Note, that these callbacks
    # do not need to be registered with the data connector.
    logger_callback = {
        "logger": class_config(LoggingCallback, refresh_rate=50)
    }
    ckpt_callback = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.output_dir,
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

    # Assign the defined callbacks to the config
    config.shared_callbacks = {
        **logger_callback,
        **eval_callbacks,
    }

    config.train_callbacks = {
        **ckpt_callback,
    }
    config.test_callbacks = {
        # **vis_callbacks,
    }

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    pl_trainer = ConfigDict()
    config.pl_trainer = pl_trainer

    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()


def get_sweep() -> ConfigDict:
    """Returns the config dict for a grid search over learning rate.

    Returns:
        ConfigDict: The configuration that can be used to run a grid search.
            It can be passed to replicate_config to create a list of configs
            that can be used to run a grid search.
    """
    return linear_grid_search("params.lr", 0.001, 0.01, 3)
