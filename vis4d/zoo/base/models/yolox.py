"""YOLOX base model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD

from vis4d.config import class_config
from vis4d.config.typing import OptimizerConfig
from vis4d.data.const import CommonKeys as K
from vis4d.engine.callbacks import (
    EMACallback,
    YOLOXModeSwitchCallback,
    YOLOXSyncNormCallback,
    YOLOXSyncRandomResizeCallback,
)
from vis4d.engine.connectors import LossConnector, data_key, pred_key
from vis4d.engine.loss_module import LossModule
from vis4d.engine.optim.scheduler import ConstantLR, QuadraticLRWarmup
from vis4d.model.adapter import ModelExpEMAAdapter
from vis4d.model.detect.yolox import YOLOX
from vis4d.op.base import CSPDarknet
from vis4d.op.detect.yolox import YOLOXHead, YOLOXHeadLoss
from vis4d.op.fpp import YOLOXPAFPN
from vis4d.zoo.base import get_lr_scheduler_cfg, get_optimizer_cfg

# Data connectors
CONN_YOLOX_LOSS_2D = {
    "cls_outs": pred_key("cls_score"),
    "reg_outs": pred_key("bbox_pred"),
    "obj_outs": pred_key("objectness"),
    "target_boxes": data_key(K.boxes2d),
    "target_class_ids": data_key(K.boxes2d_classes),
    "images_hw": data_key(K.input_hw),
}


def get_yolox_optimizers_cfg(
    lr: float | FieldReference,
    num_epochs: int | FieldReference,
    warmup_epochs: int,
    num_last_epochs: int,
) -> list[OptimizerConfig]:
    """Construct optimizer for YOLOX training."""
    return [
        get_optimizer_cfg(
            optimizer=class_config(
                SGD,
                lr=lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            ),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(QuadraticLRWarmup, max_steps=warmup_epochs),
                    end=warmup_epochs,
                    epoch_based=False,
                    convert_epochs_to_steps=True,
                    convert_attributes=["max_steps"],
                ),
                get_lr_scheduler_cfg(
                    class_config(
                        CosineAnnealingLR,
                        T_max=num_epochs - num_last_epochs - warmup_epochs,
                        eta_min=lr * 0.05,
                    ),
                    begin=warmup_epochs,
                    end=num_epochs - num_last_epochs,
                    epoch_based=False,
                    convert_epochs_to_steps=True,
                    convert_attributes=["T_max"],
                ),
                get_lr_scheduler_cfg(
                    class_config(
                        ConstantLR, max_steps=num_last_epochs, factor=1.0
                    ),
                    begin=num_epochs - num_last_epochs,
                    end=num_epochs,
                    epoch_based=True,
                ),
            ],
            param_groups=[
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


def get_yolox_callbacks_cfg(
    switch_epoch: int,
    shape: tuple[int, int] = (480, 480),
    num_sizes: int = 11,
    use_ema: bool = True,
) -> list[ConfigDict]:
    """Get YOLOX callbacks for training."""
    callbacks = []
    if num_sizes > 0:
        callbacks.append(
            class_config(
                YOLOXSyncRandomResizeCallback,
                size_list=[
                    (shape[0] + i * 32, shape[1] + i * 32)
                    for i in range(num_sizes)
                ],
                interval=10,
            )
        )
    callbacks += [
        class_config(YOLOXModeSwitchCallback, switch_epoch=switch_epoch),
        class_config(YOLOXSyncNormCallback),
    ]
    if use_ema:
        callbacks += [class_config(EMACallback)]
    return callbacks


def get_model_setting(model_type: str) -> tuple[float, float, int, list[int]]:
    """Get YOLOX model setting."""
    if model_type == "tiny":
        deepen_factor, widen_factor, num_csp_blocks = 0.33, 0.375, 1
        in_channels = [96, 192, 384]
    elif model_type == "small":
        deepen_factor, widen_factor, num_csp_blocks = 0.33, 0.5, 1
        in_channels = [128, 256, 512]
    elif model_type == "large":
        deepen_factor, widen_factor, num_csp_blocks = 1.0, 1.0, 3
        in_channels = [256, 512, 1024]
    elif model_type == "xlarge":
        deepen_factor, widen_factor, num_csp_blocks = 1.33, 1.25, 4
        in_channels = [320, 640, 1280]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return deepen_factor, widen_factor, num_csp_blocks, in_channels


def get_yolox_model_cfg(
    num_classes: FieldReference | int, model_type: str
) -> ConfigDict:
    """Get YOLOX model."""
    assert model_type in {"tiny", "small", "large", "xlarge"}, (
        f"model_type must be one of 'tiny', 'small', 'large', 'xlarge', "
        f"got {model_type}."
    )
    (
        deepen_factor,
        widen_factor,
        num_csp_blocks,
        in_channels,
    ) = get_model_setting(model_type)
    basemodel = class_config(
        CSPDarknet, deepen_factor=deepen_factor, widen_factor=widen_factor
    )
    fpn = class_config(
        YOLOXPAFPN,
        in_channels=in_channels,
        out_channels=in_channels[0],
        num_csp_blocks=num_csp_blocks,
    )
    yolox_head = class_config(
        YOLOXHead,
        num_classes=num_classes,
        in_channels=in_channels[0],
        feat_channels=in_channels[0],
    )
    return basemodel, fpn, yolox_head


def get_yolox_cfg(
    num_classes: FieldReference | int,
    model_type: str,
    use_ema: bool = True,
    weights: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Return default config for YOLOX model and loss.

    Args:
        num_classes (FieldReference | int): Number of classes.
        model_type (str): Model type. Must be one of 'tiny', 'small', 'large',
            'xlarge'.
        use_ema (bool, optional): Whether to use EMA. Defaults to True.
        weights (str | None, optional): Weights to load. Defaults to None.
    """
    ######################################################
    ##                        MODEL                     ##
    ######################################################
    basemodel, fpn, yolox_head = get_yolox_model_cfg(num_classes, model_type)
    model = class_config(
        YOLOX,
        num_classes=num_classes,
        basemodel=basemodel,
        fpn=fpn,
        yolox_head=yolox_head,
        weights=weights,
    )
    if use_ema:
        model = class_config(ModelExpEMAAdapter, model=model)

    ######################################################
    ##                      LOSS                        ##
    ######################################################
    loss = class_config(
        LossModule,
        losses=[
            {
                "loss": class_config(YOLOXHeadLoss, num_classes=num_classes),
                "connector": class_config(
                    LossConnector, key_mapping=CONN_YOLOX_LOSS_2D
                ),
            },
        ],
    )
    return model, loss
