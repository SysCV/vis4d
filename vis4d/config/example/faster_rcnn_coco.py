"""Faster RCNN COCO training example."""
from __future__ import annotations

import warnings

from ml_collections import ConfigDict

from vis4d.config.default.connectors import default_detection_connector
from vis4d.config.default.loss.faster_rcnn_loss import (
    get_default_faster_rcnn_loss,
)
from vis4d.engine.connectors import SourceKeyDescription
from vis4d.model.detect.faster_rcnn import FasterRCNN

warnings.filterwarnings("ignore")
import os

from vis4d.config.default.data.dataloader import default_dataloader_config
from vis4d.config.default.data.detect import default_detection_preprocessing
from vis4d.config.util import class_config
from vis4d.data.datasets.coco import COCO

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
    engine.learning_rate = 0.02 / 16 * engine.get_ref("batch_size")
    engine.experiment_name = "frcnn_coco_epoch"
    engine.save_prefix = "vis4d-workspace/test/" + engine.get_ref(
        "experiment_name"
    )
    engine.metric = "COCO_AP"
    engine.num_epochs = 10
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
        # FIXME: Currently, resolving transforms is broken if we directly pass
        # the function instead of the name to resolve, since resolving
        # the function path with the decorator converts e.g. 'pad_image' which
        # is 'BatchTransform.__call__.<locals>.get_transform_fn'
        # to  vis4d.data.transforms.base.get_transform_fn.
        # We need to use to full config path for now. Should probably be fixed
        # with the transform update
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
    ##                  DATA CONNECTOR                  ##
    ######################################################

    coco_eval = dict()
    coco_eval["coco_image_id"] = SourceKeyDescription(
        key="coco_image_id", source="data"
    )
    coco_eval["pred_boxes"] = SourceKeyDescription(
        key="pred_boxes", source="prediction"
    )
    coco_eval["pred_scores"] = SourceKeyDescription(
        key="pred_scores", source="prediction"
    )
    coco_eval["pred_classes"] = SourceKeyDescription(
        key="pred_classes", source="prediction"
    )

    data_connector_cfg = default_detection_connector(dict(coco=coco_eval))

    config.data_connector = data_connector_cfg

    return config
