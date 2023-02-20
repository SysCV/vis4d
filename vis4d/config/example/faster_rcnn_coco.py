"""Faster RCNN COCO training example."""
from __future__ import annotations

import os

from torch import optim
from torch.optim.lr_scheduler import StepLR

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.config.default.data.dataloader import default_image_dl
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
from vis4d.vis.image import BoundingBoxVisualizer

# This is just for demo purposes. Uses the relative path to the vis4d root.
VIS4D_ROOT = os.path.abspath(os.path.dirname(__file__) + "../../../../")
COCO_DATA_ROOT = os.path.join(VIS4D_ROOT, "tests/vis4d-test-data/coco_test")
TRAIN_SPLIT = "train"
TEST_SPLIT = "train"  # "val"


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
    config.experiment_name = "frcnn_coco"
    config.save_prefix = "vis4d-workspace/test/" + config.get_ref(
        "experiment_name"
    )

    config.dataset_root = COCO_DATA_ROOT
    config.train_split = TRAIN_SPLIT
    config.test_split = TEST_SPLIT
    config.n_gpus = 1
    config.num_epochs = 10

    ## High level hyper parameters
    params = ConfigDict()
    params.batch_size = 16
    params.lr = 0.01
    params.augment_proba = 0.5
    params.num_classes = 80
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
        keys_to_load=(CK.images, CK.boxes2d, CK.boxes2d_classes),
        data_root=config.dataset_root,
        split=config.train_split,
    )
    preproc = det_preprocessing(800, 1333, params.augment_proba)
    dataloader_train_cfg = default_image_dl(
        preproc, dataset_cfg_train, params.batch_size, shuffle=True
    )
    config.train_dl = dataloader_train_cfg

    # Test
    dataset_test_cfg = class_config(
        COCO,
        keys_to_load=(CK.images, CK.boxes2d, CK.boxes2d_classes),
        data_root=config.dataset_root,
        split=config.test_split,
    )
    preprocess_test_cfg = det_preprocessing(800, 1333, augment_probability=0)
    dataloader_cfg_test = default_image_dl(
        preprocess_test_cfg,
        dataset_test_cfg,
        batch_size=1,
        num_workers_per_gpu=1,
        shuffle=False,
    )
    config.test_dl = {"coco_eval": dataloader_cfg_test}

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
        weights="mmdet",
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
    # We use the default connections provided for faster_rcnn.

    config.data_connector = class_config(
        StaticDataConnector,
        connections=DataConnectionInfo(
            train=CONN_BBOX_2D_TRAIN,
            test=CONN_BBOX_2D_TEST,
            loss={**CONN_RPN_LOSS_2D, **CONN_ROI_LOSS_2D},
            callbacks={
                "coco_eval_test": CONN_COCO_BBOX_EVAL,
                "bbox_vis_test": CONN_BBOX_2D_VIS,
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


def get_sweep() -> ConfigDict:
    """Returns the config dict for a grid search over learning rate.

    Returns:
        ConfigDict: The configuration that can be used to run a grid search.
            It can be passed to replicate_config to create a list of configs
            that can be used to run a grid search.
    """
    return linear_grid_search("params.lr", 0.001, 0.01, 3)
