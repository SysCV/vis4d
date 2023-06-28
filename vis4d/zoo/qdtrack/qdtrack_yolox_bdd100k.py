# pylint: disable=duplicate-code
"""QDTrack-YOLOX BDD100K."""
from __future__ import annotations

import pytorch_lightning as pl
from ml_collections import ConfigDict
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from vis4d.config import FieldConfigDict, class_config
from vis4d.config.common.types import ExperimentConfig, ExperimentParameters
from vis4d.config.default import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
)
from vis4d.config.default.data_connectors import CONN_BBOX_2D_TRACK_VIS
from vis4d.config.util import (
    get_callable_cfg,
    get_inference_dataloaders_cfg,
    get_optimizer_cfg,
    get_train_dataloader_cfg,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.bdd100k import BDD100K, bdd100k_track_map
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.reference import MultiViewDataset, UniformViewSampler
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropImages,
    GenCropParameters,
)
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImages
from vis4d.data.transforms.mosaic import (
    GenMosaicParameters,
    MosaicBoxes2D,
    MosaicImages,
)
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.photometric import ColorJitter
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
    data_key,
    pred_key,
)
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim.warmup import QuadraticLRWarmup
from vis4d.eval.bdd100k import BDD100KTrackEvaluator
from vis4d.model.track.qdtrack import YOLOXQDTrack
from vis4d.op.loss.common import smooth_l1_loss
from vis4d.op.track.qdtrack import QDTrackInstanceSimilarityLoss
from vis4d.vis.image import BoundingBoxVisualizer

CONN_BBOX_2D_TRAIN = {
    "images": K.images,
    "images_hw": K.input_hw,
    "frame_ids": K.frame_ids,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "boxes2d_track_ids": K.boxes2d_track_ids,
    "keyframes": "keyframes",
}

CONN_BBOX_2D_TEST = {
    "images": K.images,
    "images_hw": K.input_hw,
    "original_hw": K.original_hw,
    "frame_ids": K.frame_ids,
}

CONN_BDD100K_EVAL = {
    "frame_ids": data_key("frame_ids"),
    "sample_names": data_key("sample_names"),
    "sequence_names": data_key("sequence_names"),
    "pred_boxes": pred_key("boxes"),
    "pred_classes": pred_key("class_ids"),
    "pred_scores": pred_key("scores"),
    "pred_track_ids": pred_key("track_ids"),
}

CONN_TRACK_LOSS_2D = {
    "key_embeddings": pred_key("key_embeddings"),
    "ref_embeddings": pred_key("ref_embeddings"),
    "key_track_ids": pred_key("key_track_ids"),
    "ref_track_ids": pred_key("ref_track_ids"),
}


def get_train_dataloader(
    data_backend: None | ConfigDict,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default train dataloader for BDD100K tracking."""
    bdd100k_det_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/100k/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/det_20/det_train.json",
        category_map=bdd100k_track_map,
        config_path="det",
        image_channel_mode="BGR",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/det_train.pkl",
    )

    bdd100k_track_train = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track/train/",
        keys_to_load=(K.images, K.boxes2d),
        annotation_path="data/bdd100k/labels/box_track_20/train/",
        category_map=bdd100k_track_map,
        config_path="box_track",
        image_channel_mode="BGR",
        data_backend=data_backend,
        skip_empty_samples=True,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_train.pkl",
    )

    train_dataset_cfg = [
        class_config(
            MultiViewDataset,
            dataset=bdd100k_det_train,
            sampler=class_config(
                UniformViewSampler, scope=0, num_ref_samples=1
            ),
        ),
        class_config(
            MultiViewDataset,
            dataset=bdd100k_track_train,
            sampler=class_config(
                UniformViewSampler, scope=3, num_ref_samples=1
            ),
        ),
    ]

    preprocess_transforms = [
        class_config(GenMosaicParameters, out_shape=(800, 1440)),
        class_config(MosaicImages),
        class_config(MosaicBoxes2D),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[class_config(FlipImages), class_config(FlipBoxes2D)],
            probability=0.5,
        )
    )

    preprocess_transforms += [
        class_config(
            GenResizeParameters,
            shape=(800, 1440),
            scale_range=(0.5, 1.5),
            keep_ratio=True,
        ),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms += [
        class_config(GenCropParameters, shape=(800, 1440)),
        class_config(CropImages),
        class_config(CropBoxes2D),
    ]

    preprocess_transforms.append(class_config(ColorJitter))

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(PadImages), class_config(ToTensor)]
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=train_batchprocess_cfg,
    )


def get_test_dataloader(
    data_backend: None | ConfigDict,
    samples_per_gpu: int,
    workers_per_gpu: int,
) -> ConfigDict:
    """Get the default test dataloader for BDD100K tracking."""
    test_dataset = class_config(
        BDD100K,
        data_root="data/bdd100k/images/track/val/",
        keys_to_load=(K.images, K.original_images),
        annotation_path="data/bdd100k/labels/box_track_20/val/",
        category_map=bdd100k_track_map,
        config_path="box_track",
        image_channel_mode="BGR",
        data_backend=data_backend,
        cache_as_binary=True,
        cached_file_path="data/bdd100k/pkl/track_val.pkl",
    )

    preprocess_transforms = [
        class_config(
            GenResizeParameters,
            shape=(800, 1440),
            keep_ratio=False,
            align_long_edge=True,
        ),
        class_config(ResizeImages),
    ]

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(PadImages), class_config(ToTensor)]
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        video_based_inference=True,
        batchprocess_cfg=test_batchprocess_cfg,
    )


def get_config() -> FieldConfigDict:
    """Returns the config dict for qdtrack on bdd100k.

    Returns:
        ConfigDict: The configuration
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="qdtrack_yolox_bdd100k")

    ckpt_path = (
        "vis4d-workspace/QDTrack/pretrained/qdtrack-yolox-ema_bdd100k.ckpt"
    )

    # Hyper Parameters
    params = FieldConfigDict()
    params.samples_per_gpu = 2
    params.workers_per_gpu = 2
    params.lr = 0.01
    params.num_epochs = 12
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data = FieldConfigDict()
    data_backend = class_config(HDF5Backend)

    data.train_dataloader = get_train_dataloader(
        data_backend, params.samples_per_gpu, params.workers_per_gpu
    )
    data.test_dataloader = get_test_dataloader(data_backend, 1, 1)

    config.data = data

    ######################################################
    ##                        MODEL                     ##
    ######################################################
    num_classes = len(bdd100k_track_map)

    config.model = class_config(
        YOLOXQDTrack, num_classes=num_classes, weights=ckpt_path
    )

    ######################################################
    ##                        LOSS                      ##
    ######################################################
    # rcnn_box_encoder, _ = get_default_rcnn_box_codec_cfg()

    # rcnn_loss = class_config(
    #     RCNNLoss,
    #     box_encoder=rcnn_box_encoder,
    #     num_classes=num_classes,
    #     loss_bbox=get_callable_cfg(smooth_l1_loss),
    # )

    track_loss = class_config(QDTrackInstanceSimilarityLoss)

    config.loss = class_config(
        LossModule,
        losses=[
            # {
            #     "loss": rcnn_loss,
            #     "connector": class_config(
            #         LossConnector, key_mapping=CONN_ROI_LOSS_2D
            #     ),
            # },
            {
                "loss": track_loss,
                "connector": class_config(
                    LossConnector, key_mapping=CONN_TRACK_LOSS_2D
                ),
            },
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD,
                lr=params.lr,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True,
            ),
            lr_scheduler=class_config(
                CosineAnnealingLR, T_max=999, eta_min=params.lr * 0.05
            ),
            lr_warmup=class_config(
                QuadraticLRWarmup, warmup_ratio=1.0, warmup_steps=1000
            ),
            epoch_based_lr=True,
            epoch_based_warmup=False,
            param_groups_cfg=[
                {
                    "custom_keys": ["basemodel", "fpn", "yolox_head"],
                    "norm_decay_mult": 0.0,
                },
                {
                    "custom_keys": ["basemodel", "fpn", "yolox_head"],
                    "bias_decay_mult": 0.0,
                },
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config)

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer, vis_freq=500),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_2D_TRACK_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                BDD100KTrackEvaluator,
                annotation_path="data/bdd100k/labels/box_track_20/val/",
            ),
            save_prefix=config.output_dir,
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BDD100K_EVAL
            ),
        ),
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.precision = "16-mixed"
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.callbacks.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
