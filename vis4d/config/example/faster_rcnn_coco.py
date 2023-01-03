"""Faster RCNN COCO training example."""
from __future__ import annotations

import warnings

from ml_collections import ConfigDict

from vis4d.config.default.connectors import default_detection_connector
from vis4d.data.connectors import SourceKeyDescription

warnings.filterwarnings("ignore")
from tests.util import get_test_data
from vis4d.config.default.data.dataloader import default_dataloader_config
from vis4d.config.default.data.detect import default_detection_preprocessing
from vis4d.config.util import class_config

COCO_DATA_ROOT = get_test_data("coco_test")
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
        "vis4d.data.datasets.coco.COCO",
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
    )

    # Test
    dataset_test_cfg = class_config(
        "vis4d.data.datasets.coco.COCO",
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
    )

    config.train_dl = dataloader_train_cfg
    config.test_dl = [dataloader_cfg_test]

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
