"""Semantic FPN BDD100K training example."""
from __future__ import annotations

from torch import optim

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    LoggingCallback,
)
from vis4d.config.default.data.dataloader import default_image_dataloader
from vis4d.config.default.data.segment import segment_preprocessing
from vis4d.config.default.data_connectors.segment import (
    CONN_BDD100K_SEGMENT_EVAL,
    CONN_MASKS_TEST,
    CONN_MASKS_TRAIN,
    CONN_SEGMENT_LOSS,
)
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.bdd100k import BDD100K
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.connectors import DataConnectionInfo, StaticDataConnector
from vis4d.eval.segment.bdd100k import BDD100KSemSegEvaluator
from vis4d.model.segment.semantic_fpn import SemanticFPN, SemanticFPNLoss
from vis4d.optim import PolyLR
from vis4d.optim.warmup import LinearLRWarmup


def get_config() -> ConfigDict:
    """Returns the config dict for the BDD100K semantic segmentation task.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################

    config = ConfigDict()
    config.n_gpus = 1
    config.work_dir = "vis4d-workspace"
    config.experiment_name = "test/semantic_fpn_bdd100k"
    config = set_output_dir(config)

    config.dataset_root = "./data/bdd100k/images/10k"
    config.train_split = "train"
    config.test_split = "val"

    ## High level hyper parameters
    params = ConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_steps = 40000
    params.num_epochs = 45
    params.augment_prob = 0.5
    params.num_classes = 19
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################

    data = ConfigDict()
    data_backend = HDF5Backend()

    # Training Datasets
    dataset_cfg_train = class_config(
        BDD100K,
        data_root=f"data/bdd100k/images/10k/train",
        annotation_path=f"data/bdd100k/labels/sem_seg_train_rle.json",
        config_path="sem_seg",
        keys_to_load=(K.images, K.segmentation_masks),
        data_backend=data_backend,
    )
    preproc = segment_preprocessing(720, 1280, True, params.augment_prob)
    dataloader_train_cfg = default_image_dataloader(
        preprocess_cfg=preproc,
        dataset_cfg=dataset_cfg_train,
        num_samples_per_gpu=params.samples_per_gpu,
        num_workers_per_gpu=params.workers_per_gpu,
        shuffle=True,
    )
    data.train_dataloader = dataloader_train_cfg

    # Test
    dataset_test_cfg = class_config(
        BDD100K,
        data_root=f"data/bdd100k/images/10k/val",
        annotation_path=f"data/bdd100k/labels/sem_seg_val_rle.json",
        config_path="sem_seg",
        keys_to_load=(K.images, K.segmentation_masks),
        data_backend=data_backend,
    )
    preprocess_test_cfg = segment_preprocessing(
        720, 1280, True, augment_probability=0
    )
    dataloader_cfg_test = default_image_dataloader(
        preprocess_cfg=preprocess_test_cfg,
        dataset_cfg=dataset_test_cfg,
        num_samples_per_gpu=1,
        num_workers_per_gpu=1,
        shuffle=False,
        train=False,
    )
    data.test_dataloader = {"bdd100k_eval": dataloader_cfg_test}

    config.data = data

    ######################################################
    ##                   MODEL & LOSS                   ##
    ######################################################

    config.model = class_config(SemanticFPN, num_classes=params.num_classes)
    config.loss = class_config(SemanticFPNLoss)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################

    config.optimizers = [
        optimizer_cfg(
            optimizer=class_config(
                optim.SGD, lr=params.lr, momentum=0.9, weight_decay=0.0005
            ),
            lr_scheduler=class_config(
                PolyLR, max_steps=params.num_steps, min_lr=0.0001, power=0.9
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=0.001, warmup_steps=500
            ),
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################

    config.data_connector = class_config(
        StaticDataConnector,
        connections=DataConnectionInfo(
            train=CONN_MASKS_TRAIN,
            test=CONN_MASKS_TEST,
            loss=CONN_SEGMENT_LOSS,
            callbacks={"bdd100k_eval": CONN_BDD100K_SEGMENT_EVAL},
        ),
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################

    eval_callbacks = {
        "bdd100k_eval": class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KSemSegEvaluator,
                annotation_path=f"data/bdd100k/labels/sem_seg_val_rle.json",
            ),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        )
    }

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
    config.shared_callbacks = {**logger_callback}

    config.train_callbacks = {**ckpt_callback}
    config.test_callbacks = {**eval_callbacks}

    return config.value_mode()
