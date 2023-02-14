"""Faster RCNN COCO training example."""
from __future__ import annotations

import warnings
from sys import base_prefix

from ml_collections import ConfigDict

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.config.default.data_connectors.faster_rcnn import (
    eval_bbox_conn,
    loss_conn,
    test_data_conn,
    train_data_conn,
    vis_bbox_conn,
)
from vis4d.config.default.loss.faster_rcnn_loss import (
    get_default_faster_rcnn_loss,
)
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.engine.connectors import DataConnectionInfo
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.model.detect.faster_rcnn import FasterRCNN

warnings.filterwarnings("ignore")
import os

from torch import optim
from torch.optim.lr_scheduler import StepLR

from vis4d.config.default.data.dataloader import default_image_dl
from vis4d.config.default.data.detect import default_detection_preprocessing
from vis4d.config.util import class_config
from vis4d.data.datasets.coco import COCO
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
    >>> python -m vis4d.run --config vis4d/config/example/faster_rcnn_coco_simple.py --config.params.num_epochs 100 -- config.params.lr 0.001

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

    ## High level hyper parameters
    params = ConfigDict()
    params.batch_size = 16
    params.lr = 0.01
    params.num_epochs = 0.01
    params.n_gpus = 0.01
    params.augmentation_probability = 0.5
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
        data_root=config.get_ref("dataset_root"),
        split=config.get_ref("train_split"),
    )
    preprocess_cfg_train = default_detection_preprocessing(
        800, 1333, augmentation_probability=params.augmentation_probability
    )

    dataloader_train_cfg = default_image_dl(
        preprocess_cfg_train,
        dataset_cfg_train,
        params.get_ref("batch_size"),
        shuffle=True,
    )
    config.train_dl = dataloader_train_cfg

    # Test
    dataset_test_cfg = class_config(
        COCO,
        data_root=config.get_ref("dataset_root"),
        split=config.get_ref("test_split"),
    )
    preprocess_test_cfg = default_detection_preprocessing(
        800, 1333, augmentation_probability=0.0
    )
    dataloader_cfg_test = default_image_dl(
        preprocess_test_cfg,
        dataset_test_cfg,
        batch_size=1,
        num_workers_per_gpu=1,
        shuffle=False,
    )
    config.test_dl = [dataloader_cfg_test]

    ######################################################
    ##                        MODEL                     ##
    ######################################################

    # Here we define the model. We use the default Faster RCNN model
    # provided by vis4d.

    config.model = class_config(
        FasterRCNN, num_classes=params.get_ref("num_classes")
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    # Here we define the loss function. We use the default loss function
    # provided for the Faster RCNN model.
    # Note, that the loss functions consists of multiple loss terms which
    # are averaged using a weighted sum.

    config.loss = class_config(get_default_faster_rcnn_loss)

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
    #           fun=class_config(optim.SGD, lr=params.get_ref("lr"))
    #        )
    #    )
    # ]

    config.optimizers = [
        optimizer_cfg(
            optimizer=class_config(optim.SGD, lr=params.get_ref("lr")),
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

    config.data_connector = DataConnectionInfo(
        train=train_data_conn(),
        test=test_data_conn(),
        loss=loss_conn(),
        evaluators={"coco": eval_bbox_conn()},
        vis={"coco": vis_bbox_conn()},
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################

    # Here we define the evaluator. We use the default COCO evaluator for
    # bounding box detection. Note, that we need to define the connections
    # between the evaluator and the data connector in the data connector
    # section. And use the same name here.

    eval_callbacks = {
        "coco": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCOEvaluator,
                data_root=config.get_ref("dataset_root"),
                split=config.get_ref("test_split"),
            ),
            test_every_nth_epoch=1,
            num_epochs=params.get_ref("num_epochs"),
            eval_connector=config.get_ref("data_connector"),
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
        "bbox": class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer),
            vis_every_nth_epoch=1,
            num_epochs=params.get_ref("num_epochs"),
            output_dir=os.path.join(config.get_ref(base_prefix), "vis"),
            data_connector=config.get_ref("data_connector"),
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
            save_prefix=config.get_ref("save_prefix"),
            save_every_nth_epoch=1,
        )
    }

    # Assign the defined callbacks to the config
    config.test_callbacks = {**eval_callbacks, **vis_callbacks}

    return config
