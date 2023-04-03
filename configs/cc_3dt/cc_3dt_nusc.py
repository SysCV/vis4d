"""CC-3DT nuScenes inference example."""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.common.callbacks import (
    CheckpointCallback,
    EvaluatorCallback,
    LoggingCallback,
)
from vis4d.config.default.data.dataloader import default_image_dataloader
from vis4d.config.default.optimizer.default import optimizer_cfg
from vis4d.config.default.runtime import set_output_dir
from vis4d.config.util import ConfigDict, class_config
from vis4d.data.const import CommonKeys as CK
from vis4d.data.datasets.nuscenes import (
    NuScenes,
    nuscenes_class_range_map,
    nuscenes_track_map,
)
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.loader import VideoDataPipe, multi_sensor_collate
from vis4d.engine.connectors import (
    DataConnectionInfo,
    MultiSensorDataConnector,
    data_key,
    pred_key,
)
from vis4d.eval.track3d.nuscenes import NuScenesEvaluator
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT

from vis4d.data.transforms.base import compose, compose_batch
from vis4d.data.transforms.normalize import BatchNormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeImage,
    ResizeIntrinsics,
)

CONN_BBOX_3D_TEST = {
    CK.images: CK.images,
    CK.original_hw: "images_hw",
    CK.intrinsics: CK.intrinsics,
    CK.extrinsics: CK.extrinsics,
    CK.frame_ids: CK.frame_ids,
}

CONN_NUSC_EVAL = {
    "token": data_key("token"),
    "boxes_3d": pred_key("boxes_3d"),
    "class_ids": pred_key("class_ids"),
    "scores_3d": pred_key("scores_3d"),
    "track_ids": pred_key("track_ids"),
}


def get_config() -> ConfigDict:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = ConfigDict()
    config.n_gpus = 8
    config.work_dir = "vis4d-workspace"
    config.experiment_name = "cc_3dt_r50_kf3d"
    config = set_output_dir(config)

    ckpt_path = "https://dl.cv.ethz.ch/vis4d/cc_3dt_R_50_FPN_nuscenes.pt"

    # Hyper Parameters
    params = ConfigDict()
    params.samples_per_gpu = 4
    params.lr = 0.01
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = ConfigDict()
    dataset_root = "data/nuscenes"
    version = "v1.0-trainval"
    train_split = "train"
    test_split = "val"
    metadata = ["use_camera"]
    data_backend = HDF5Backend()

    # TODO: Add train dataset
    train_dataset_cfg = None
    train_dataloader_cfg = None
    data.train_dataloader = {"nusc_train": train_dataloader_cfg}

    # Test
    test_dataset_cfg = class_config(
        NuScenes,
        data_root=dataset_root,
        version=version,
        split=test_split,
        metadata=metadata,
        data_backend=data_backend,
    )

    test_preprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(
                GenerateResizeParameters,
                shape=(900, 1600),
                keep_ratio=True,
                sensors=NuScenes._CAMERAS,
            ),
            class_config(
                ResizeImage,
                sensors=NuScenes._CAMERAS,
            ),
            class_config(
                ResizeIntrinsics,
                sensors=NuScenes._CAMERAS,
            ),
        ],
    )

    test_batchprocess_cfg = class_config(
        compose_batch,
        transforms=[
            class_config(
                PadImages,
                sensors=NuScenes._CAMERAS,
            ),
            class_config(
                BatchNormalizeImages,
                sensors=NuScenes._CAMERAS,
            ),
            class_config(
                ToTensor,
                sensors=NuScenes._CAMERAS,
            ),
        ],
    )

    test_dataloader_cfg = default_image_dataloader(
        test_preprocess_cfg,
        test_dataset_cfg,
        num_samples_per_gpu=1,
        batchprocess_cfg=test_batchprocess_cfg,
        data_pipe=VideoDataPipe,
        train=False,
        collate_fn=multi_sensor_collate,
    )
    data.test_dataloader = {"nusc_eval": test_dataloader_cfg}
    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    backbone = "resnet50"
    motion_model = "KF3D"
    pure_det = False
    num_classes = len(nuscenes_track_map)
    class_range_map = torch.Tensor(nuscenes_class_range_map)

    config.model = class_config(
        FasterRCNNCC3DT,
        num_classes=num_classes,
        backbone=backbone,
        motion_model=motion_model,
        pure_det=pure_det,
        class_range_map=class_range_map,
        weights=ckpt_path,
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    config.loss = None  # TODO: implement loss

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        optimizer_cfg(
            optimizer=class_config(optim.SGD, lr=params.lr),
            lr_scheduler=class_config(
                MultiStepLR, milestones=[8, 11], gamma=0.1
            ),
            lr_warmup=None,
            epoch_based_lr=True,
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
        MultiSensorDataConnector,
        connections=DataConnectionInfo(
            test=CONN_BBOX_3D_TEST,
            callbacks={"nusc_eval_test": CONN_NUSC_EVAL},
        ),
        default_sensor=NuScenes._CAMERAS[0],
        sensors=NuScenes._CAMERAS,
    )

    ######################################################
    ##                     EVALUATOR                    ##
    ######################################################
    # Here we define the evaluator. We need to define the connections
    # between the evaluator and the data connector in the data connector
    # section. And use the same name here.
    eval_callbacks = {
        "nusc_eval": class_config(
            EvaluatorCallback,
            save_prefix=config.output_dir,
            evaluator=class_config(NuScenesEvaluator),
            run_every_nth_epoch=1,
            num_epochs=params.num_epochs,
        ),
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
    config.shared_callbacks = {**logger_callback, **eval_callbacks}

    config.train_callbacks = {**ckpt_callback}

    config.test_callbacks = {}

    ######################################################
    ##                  PL CALLBACKS                    ##
    ######################################################
    pl_trainer = ConfigDict()

    pl_callbacks: list[pl.callbacks.Callback] = []

    config.pl_trainer = pl_trainer
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
