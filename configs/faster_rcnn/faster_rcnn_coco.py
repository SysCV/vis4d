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
from vis4d.config.base.datasets.coco_detection import (
    det_preprocessing,
    CONN_COCO_BBOX_EVAL,
)
from vis4d.config.default.dataloader import get_dataloader_config
from vis4d.config.base.models.faster_rcnn import (
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_TRAIN,
    CONN_ROI_LOSS_2D,
    CONN_RPN_LOSS_2D,
    get_default_faster_rcnn_loss,
)
from vis4d.config.default.data_connectors import CONN_BBOX_2D_VIS
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.default.runtime import get_runtime_config
from vis4d.config.default.sweep.default import linear_grid_search
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.coco import COCO
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.model.detect.faster_rcnn import FasterRCNN
from vis4d.op.detect.faster_rcnn import (
    get_default_anchor_generator,
    get_default_rcnn_box_encoder,
    get_default_rpn_box_encoder,
)
from vis4d.optim.warmup import LinearLRWarmup


def get_config() -> ConfigDict:
    """Returns the config dict for the coco detection task.

    This is a simple example that shows how to set up a training experiment
    for the COCO detection task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.cli --config configs/faster_rcnn/faster_rcnn_coco.py --config.params.lr 0.001

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()

    config.work_dir = "vis4d-workspace"
    config.experiment_name = "faster_rcnn_r50_fpn_coco"
    config = get_runtime_config(config)

    config.dataset_root = "data/coco"
    config.train_split = "train2017"
    config.test_split = "val2017"

    # High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.lr = 0.02
    params.num_epochs = 12
    params.augment_proba = 0.5
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = ConfigDict()
    data_backend = HDF5Backend()

    # Train
    train_dataset_cfg = class_config(
        COCO,
        keys_to_load=(K.images, K.boxes2d, K.boxes2d_classes),
        data_root=config.dataset_root,
        split=config.train_split,
        remove_empty=True,
        data_backend=data_backend,
    )
    train_preprocess_cfg = det_preprocessing(800, 1333, params.augment_proba)
    data.train_dataloader = get_dataloader_config(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        num_samples_per_gpu=params.samples_per_gpu,
        shuffle=True,
    )

    # Test
    test_dataset_cfg = class_config(
        COCO,
        keys_to_load=(K.images, K.boxes2d, K.boxes2d_classes),
        data_root=config.dataset_root,
        split=config.test_split,
        data_backend=data_backend,
    )
    test_preprocess_cfg = det_preprocessing(800, 1333, augment_probability=0)
    data.test_dataloader = get_dataloader_config(
        preprocess_cfg=test_preprocess_cfg,
        dataset_cfg=test_dataset_cfg,
        num_samples_per_gpu=1,
        train=False,
    )

    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    # Here we define the model. We use the default Faster RCNN model
    # provided by vis4d.
    anchor_generator = class_config(get_default_anchor_generator)
    rcnn_box_encoder = class_config(get_default_rcnn_box_encoder)
    rpn_box_encoder = class_config(get_default_rpn_box_encoder)

    config.model = class_config(
        FasterRCNN,
        # weights="mmdet",
        num_classes=params.num_classes,
        rpn_box_encoder=rpn_box_encoder,
        rcnn_box_encoder=rcnn_box_encoder,
        anchor_generator=anchor_generator,
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
        rpn_box_encoder=rpn_box_encoder,
        rcnn_box_encoder=rcnn_box_encoder,
        anchor_generator=anchor_generator,
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_config(
            optimizer=class_config(
                optim.SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
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
    # pl_trainer.wandb = True
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
