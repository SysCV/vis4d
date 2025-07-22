# pylint: disable=duplicate-code
"""RetinaNet COCO training example."""
from __future__ import annotations

import lightning.pytorch as pl
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from torch.optim.sgd import SGD

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
)
from vis4d.engine.loss_module import LossModule
from vis4d.eval.coco import COCODetectEvaluator
from vis4d.model.detect.retinanet import RetinaNet
from vis4d.op.box.encoder import DeltaXYWHBBoxEncoder
from vis4d.op.detect.retinanet import (
    RetinaNetHeadLoss,
    get_default_anchor_generator,
)
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.data_connectors import (
    CONN_BBOX_2D_VIS,
    CONN_BOX_LOSS_2D,
    CONN_IMAGES_TEST,
    CONN_IMAGES_TRAIN,
)
from vis4d.zoo.base.datasets.coco import (
    CONN_COCO_BBOX_EVAL,
    get_coco_detection_cfg,
)


def get_config() -> ExperimentConfig:
    """Returns the RetinaNet config dict for the coco detection task.

    This is an example that shows how to set up a training experiment for the
    COCO detection task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.run fit --config vis4d/zoo/retinanet/retinanet_rcnn_coco.py --config.num_epochs 100 --config.params.lr 0.001

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="retinanet_r50_fpn_coco")

    # High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_epochs = 12
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/coco"
    train_split = "train2017"
    test_split = "val2017"

    data_backend = class_config(HDF5Backend)

    config.data = get_coco_detection_cfg(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model = class_config(
        RetinaNet,
        num_classes=params.num_classes,
        # weights="mmdet",
    )

    box_encoder = class_config(
        DeltaXYWHBBoxEncoder,
        target_means=(0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0),
    )

    anchor_generator = class_config(get_default_anchor_generator)

    retina_loss = class_config(
        RetinaNetHeadLoss,
        box_encoder=box_encoder,
        anchor_generator=anchor_generator,
    )

    config.loss = class_config(
        LossModule,
        losses={
            "loss": retina_loss,
            "connector": class_config(
                LossConnector, key_mapping=CONN_BOX_LOSS_2D
            ),
        },
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(
                        LinearLR, start_factor=0.001, total_iters=500
                    ),
                    end=500,
                    epoch_based=False,
                ),
                get_lr_scheduler_cfg(
                    class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
                ),
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector,
        key_mapping=CONN_IMAGES_TRAIN,
    )

    config.test_data_connector = class_config(
        DataConnector,
        key_mapping=CONN_IMAGES_TEST,
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg()

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer, vis_freq=100),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector,
                key_mapping=CONN_BBOX_2D_VIS,
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCODetectEvaluator,
                data_root=data_root,
                split=test_split,
            ),
            metrics_to_eval=["Det"],
            test_connector=class_config(
                CallbackConnector,
                key_mapping=CONN_COCO_BBOX_EVAL,
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
