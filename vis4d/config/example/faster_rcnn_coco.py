"""Faster RCNN COCO training example."""
from __future__ import annotations

import warnings

from ml_collections import ConfigDict

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    VisualizerCallback,
)
from vis4d.config.default.connectors import default_detection_connector
from vis4d.config.default.loss.faster_rcnn_loss import (
    get_default_faster_rcnn_loss,
)
from vis4d.engine.connectors import SourceKeyDescription, data_key, pred_key
from vis4d.eval.detect.coco import COCOEvaluator
from vis4d.model.detect.faster_rcnn import FasterRCNN

warnings.filterwarnings("ignore")
import os

from torch import optim

from vis4d.config.default.data.dataloader import default_dataloader_config
from vis4d.config.default.data.detect import default_detection_preprocessing
from vis4d.config.util import class_config, delay_instantiation
from vis4d.data.datasets.coco import COCO
from vis4d.engine.opt import Optimizer
from vis4d.vis.image import BoundingBoxVisualizer

# This is just for demo purposes. Uses the relative path to the vis4d root.
VIS4D_ROOT = os.path.abspath(os.path.dirname(__file__) + "../../../../")
COCO_DATA_ROOT = os.path.join(VIS4D_ROOT, "tests/vis4d-test-data/coco_test")
TRAIN_SPLIT = "train"
TEST_SPLIT = "train"  # "val"


def get_config() -> ConfigDict:
    """Returns the config dict for the coco detection task.

    TODO, this doc. Explain fields

    Returns:
        ConfigDict: The configuration
    """
    config = ConfigDict()

    ######################################################
    ##                 Engine Information               ##
    ######################################################
    engine = ConfigDict()  # TODO rename to trainer?

    engine.batch_size = 16
    engine.lr = 0.01

    engine.learning_rate = (
        engine.get_ref("lr") / 16 * engine.get_ref("batch_size")
    )
    engine.experiment_name = "frcnn_coco"
    engine.save_prefix = "vis4d-workspace/test/" + engine.get_ref(
        "experiment_name"
    )
    engine.metric = "COCO_AP"
    engine.num_epochs = 25
    config.engine = engine

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################

    # Train
    dataset_cfg_train = class_config(
        COCO,
        data_root=COCO_DATA_ROOT,
        split=TRAIN_SPLIT,
    )
    preprocess_cfg_train = default_detection_preprocessing(
        800, 1333, augmentation_probability=0.5
    )
    dataloader_train_cfg = default_dataloader_config(
        preprocess_cfg_train,
        dataset_cfg_train,
        engine.batch_size,
        4,
        batchprocess_fn=class_config("vis4d.data.transforms.pad.pad_image"),
        # FIXME: Currently, resolving transforms is broken if we directly pass
        # the function instead of the name to resolve, since resolving
        # the function path with the decorator converts e.g. 'pad_image' which
        # is 'BatchTransform.__call__.<locals>.get_transform_fn'
        # to  vis4d.data.transforms.base.get_transform_fn.
        # We need to use to full config path for now. Should probably be fixed
        # with the transform update
        shuffle=False,
    )

    # Test
    dataset_test_cfg = class_config(
        COCO,
        data_root=COCO_DATA_ROOT,
        split=TEST_SPLIT,
    )
    preprocess_test_cfg = default_detection_preprocessing(
        800, 1333, augmentation_probability=0.0
    )
    dataloader_cfg_test = default_dataloader_config(
        preprocess_test_cfg,
        dataset_test_cfg,
        samples_per_gpu=1,
        workers_per_gpu=1,
        batchprocess_fn=class_config("vis4d.data.transforms.pad.pad_image"),
        shuffle=True,
    )

    config.train_dl = dataloader_train_cfg
    config.test_dl = [dataloader_cfg_test]

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    config.model = class_config(FasterRCNN, num_classes=80, weights=None)

    ######################################################
    ##                        LOSS                      ##
    ######################################################

    config.loss = class_config(get_default_faster_rcnn_loss)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        class_config(
            Optimizer,
            optimizer_cb=delay_instantiation(
                instantiable=class_config(
                    optim.SGD, lr=engine.get_ref("learning_rate"), momentum=0.9
                )
            ),
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################

    # data connector for evaluator
    coco_eval = {}
    coco_eval["coco_image_id"] = data_key("coco_image_id")
    coco_eval["pred_boxes"] = pred_key("boxes")
    coco_eval["pred_scores"] = pred_key("scores")
    coco_eval["pred_classes"] = pred_key("class_ids")

    eval_connectors = {"coco": coco_eval}

    # data connector for visualizer
    bbox_vis = {}
    bbox_vis["images"] = data_key("images")
    bbox_vis["boxes"] = pred_key("boxes")
    vis_connectors = {"bbox": bbox_vis}

    data_connector_cfg = default_detection_connector(
        eval_connectors, vis_connectors
    )

    config.data_connector = data_connector_cfg

    ######################################################
    ##                    CALLBACKS                     ##
    ######################################################

    config.train_callbacks = {
        "ckpt": class_config(
            CheckpointCallback,
            save_prefix=engine.get_ref("save_prefix"),
            save_every_nth_epoch=1,
        )
    }

    eval_callbacks = {
        "coco": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCOEvaluator, data_root=COCO_DATA_ROOT, split="train"
            ),
            eval_connector=config.get_ref("data_connector"),
            test_every_nth_epoch=1,
            num_epochs=engine.get_ref("num_epochs"),
        )
    }

    vis_callbacks = {
        "bbox": class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer),
            data_connector=config.get_ref("data_connector"),
            vis_every_nth_epoch=1,
            num_epochs=engine.get_ref("num_epochs"),
            output_dir=os.path.join(engine.save_prefix, "vis"),
        )
    }

    config.test_callbacks = {**eval_callbacks, **vis_callbacks}

    return config
