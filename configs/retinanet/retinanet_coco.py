"""Mask RCNN COCO training example."""
from __future__ import annotations

import pytorch_lightning as pl
from torch import optim
from torch.optim.lr_scheduler import StepLR

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.config.base.datasets.coco_detection import (
    CONN_COCO_BBOX_EVAL,
    get_coco_detection_config,
)
from vis4d.config.common.types import ExperimentConfig, ExperimentParameters
from vis4d.config.default.data_connectors import (
    CONN_BBOX_2D_VIS,
    CONN_BOX_LOSS_2D,
    CONN_IMAGES_TEST,
    CONN_IMAGES_TRAIN,
)
from vis4d.config.default.optimizer import get_optimizer_config
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.util import ConfigDict, class_config
from vis4d.engine.connectors import DataConnector
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.model.detect.retinanet import RetinaNet
from vis4d.op.detect.retinanet import (
    RetinaNetHeadLoss,
    get_default_anchor_generator,
    get_default_box_encoder,
)
from vis4d.vis.image import BoundingBoxVisualizer


def get_config() -> ExperimentConfig:
    """Returns the config dict for the coco detection task.

    This is a simple example that shows how to set up a training experiment
    for the COCO detection task.

    Note that the high level params are exposed in the config. This allows
    to easily change them from the command line.
    E.g.:
    >>> python -m vis4d.engine.cli --config vis4d/config/example/mask_rcnn_coco.py --config.num_epochs 100 --config.params.lr 0.001

    Returns:
        ExperimentConfig: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ExperimentConfig()

    config.work_dir = "vis4d-workspace"
    config.experiment_name = "retinanet_coco"
    config = set_output_dir(config)

    config.num_epochs = 10

    ## High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.augment_proba = 0.5
    params.num_classes = 80
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    config.data = get_coco_detection_config(
        image_size=(512, 512),
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    # Here we define the model. We use the default Faster RCNN model
    # provided by vis4d.
    config.gen = ConfigDict()
    config.gen.anchor_generator = class_config(get_default_anchor_generator)

    config.gen.box_encoder = class_config(get_default_box_encoder)
    config.gen.anchor_generator = class_config(get_default_anchor_generator)

    config.model = class_config(RetinaNet, num_classes=params.num_classes)

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    # Here we define the loss function. We use the default loss function
    # provided for the Faster RCNN model.
    # Note, that the loss functions consists of multiple loss terms which
    # are averaged using a weighted sum.

    config.loss = class_config(
        RetinaNetHeadLoss,
        box_encoder=config.gen.box_encoder,
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
    #    get_optimizer_config(
    #        optimizer=class_config(only_encoder_params,
    #           fun=class_config(optim.SGD, lr=params.lr"))
    #        )
    #    )
    # ]

    config.optimizers = [
        get_optimizer_config(
            optimizer=class_config(optim.SGD, lr=params.lr),
            lr_scheduler=class_config(StepLR, step_size=3, gamma=0.1),
            lr_warmup=None,
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################

    # This defines how the output of each component is connected to the next
    # component. This is a very important part of the config. It defines the
    # data flow of the pipeline.
    # We use the default connections provided for mask_rcnn. Note
    # that we build up on top of the faster_rcnn losses.
    # The faster_rcnn outputs are outputted with the key "boxes" which is why
    # we need to remap the keys of the mask_rcnn losses.
    # We do this using the remap_pred_keys function.

    config.data_connector = class_config(
        DataConnector,
        train=CONN_IMAGES_TRAIN,
        test=CONN_IMAGES_TEST,
        loss={
            **CONN_BOX_LOSS_2D,
        },
        callbacks={
            "coco_eval_test": CONN_COCO_BBOX_EVAL,
            "bbox_vis_test": CONN_BBOX_2D_VIS,
        },
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
            num_epochs=config.num_epochs,
        )
    }

    ######################################################
    ##                    VISUALIZER                    ##
    ######################################################
    # Here we define the visualizer. We use the default visualizer for
    # bounding box detection. Note, that we need to define the connections
    # between the visualizer and the data connector in the data connector
    # section. And use the same name here.

    vis_callbacks = {
        "bbox_vis": class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer),
            output_dir=config.save_prefix + "/vis",
            run_every_nth_epoch=1,
            num_epochs=config.num_epochs,
        )
    }

    ######################################################
    ##                  PL CALLBACKS                    ##
    ######################################################
    pl_trainer = ConfigDict()

    pl_callbacks: list[pl.callbacks.Callback] = []

    config.pl_trainer = pl_trainer
    config.pl_callbacks = pl_callbacks

    ######################################################
    ##                GENERIC CALLBACKS                 ##
    ######################################################
    # Here we define general, all purpose callbacks. Note, that these callbacks
    # do not need to be registered with the data connector.

    config.train_callbacks = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=config.save_prefix,
            run_every_nth_epoch=1,
            num_epochs=config.num_epochs,
        )
    }

    # Assign the defined callbacks to the config
    config.test_callbacks = {**eval_callbacks, **vis_callbacks}
    return config.value_mode()
