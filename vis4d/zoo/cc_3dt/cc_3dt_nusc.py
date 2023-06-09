"""CC-3DT nuScenes inference example."""
from __future__ import annotations

import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.config import FieldConfigDict, class_config
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.util import get_inference_dataloaders_cfg, get_optimizer_cfg
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.nuscenes import NuScenes, nuscenes_detection_range
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.loader import multi_sensor_collate
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenerateResizeParameters,
    ResizeImages,
    ResizeIntrinsics,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.callbacks import EvaluatorCallback
from vis4d.engine.connectors import (
    MultiSensorCallbackConnector,
    MultiSensorDataConnector,
    data_key,
    pred_key,
)
from vis4d.engine.optim.warmup import LinearLRWarmup
from vis4d.eval.nuscenes import NuScenesEvaluator
from vis4d.model.track3d.cc_3dt import FasterRCNNCC3DT

CONN_BBOX_3D_TEST = {
    "images": K.images,
    "images_hw": K.original_hw,
    "intrinsics": K.intrinsics,
    "extrinsics": K.extrinsics,
    "frame_ids": K.frame_ids,
}

CONN_NUSC_EVAL = {
    "tokens": data_key("token"),
    "boxes_3d": pred_key("boxes_3d"),
    "class_ids": pred_key("class_ids"),
    "scores_3d": pred_key("scores_3d"),
    "track_ids": pred_key("track_ids"),
}


def get_config() -> FieldConfigDict:
    """Returns the config dict for cc-3dt on nuScenes.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="cc_3dt_r50_kf3d")

    ckpt_path = "https://dl.cv.ethz.ch/vis4d/cc_3dt_R_50_FPN_nuscenes.pt"

    # Hyper Parameters
    params = FieldConfigDict()
    params.samples_per_gpu = 4
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = FieldConfigDict()
    dataset_root = "data/nuscenes"
    version = "v1.0-mini"
    # train_split = "mini_train"
    test_split = "mini_val"
    # version = "v1.0-trainval"
    # train_split = "train"
    # test_split = "val"
    data_backend = class_config(HDF5Backend)

    # TODO: Add train dataset
    data.train_dataloader = None

    # Test
    test_dataset = class_config(
        NuScenes,
        data_root=dataset_root,
        keys_to_load=[K.images, K.original_images, K.boxes3d],
        version=version,
        split=test_split,
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/nuscenes/mini_val.pkl",
        # cached_file_path="data/nuscenes/val.pkl",
    )

    test_preprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(
                GenerateResizeParameters,
                shape=(900, 1600),
                keep_ratio=True,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                ResizeImages,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                ResizeIntrinsics,
                sensors=NuScenes.CAMERAS,
            ),
        ],
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(
                PadImages,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                NormalizeImages,
                sensors=NuScenes.CAMERAS,
            ),
            class_config(
                ToTensor,
                sensors=NuScenes.CAMERAS,
            ),
        ],
    )

    test_dataset_cfg = class_config(
        DataPipe,
        datasets=test_dataset,
        preprocess_fn=test_preprocess_cfg,
    )

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        workers_per_gpu=params.workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
        collate_fn=multi_sensor_collate,
    )

    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    config.model = class_config(
        FasterRCNNCC3DT,
        num_classes=10,
        class_range_map=nuscenes_detection_range,
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
        get_optimizer_cfg(
            optimizer=class_config(
                SGD, lr=params.lr, momentum=0.9, weight_decay=0.0001
            ),
            lr_scheduler=class_config(
                MultiStepLR, milestones=[8, 11], gamma=0.1
            ),
            lr_warmup=class_config(
                LinearLRWarmup, warmup_ratio=0.1, warmup_steps=1000
            ),
            epoch_based_lr=True,
            epoch_based_warmup=False,
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    # TODO: Add train data connector
    config.train_data_connector = None

    config.test_data_connector = class_config(
        MultiSensorDataConnector,
        key_mapping=CONN_BBOX_3D_TEST,
        sensors=NuScenes.CAMERAS,
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config)

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(NuScenesEvaluator),
            save_predictions=True,
            save_prefix=config.output_dir,
            test_connector=class_config(
                MultiSensorCallbackConnector,
                key_mapping=CONN_NUSC_EVAL,
                sensors=NuScenes.CAMERAS,
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
