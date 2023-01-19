# type: ignore # FIXME Remove and fix this with new engine / config
"""FCN COCO training example."""
import argparse
import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from vis4d.data import DictData
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import HDF5Backend
from vis4d.engine.data.detect import (
    default_test_pipeline,
    default_train_pipeline,
)
from vis4d.engine.test import testing_loop
from vis4d.engine.trainer import training_loop
from vis4d.eval import COCOEvaluator, Evaluator
from vis4d.model.segment.fcn_resnet import FCNResNet
from vis4d.optim.warmup import LinearLRWarmup

warnings.filterwarnings("ignore")


def get_dataloaders(
    is_training: bool = False, batch_size: int = 1, num_workers: int = 1
) -> Tuple[Optional[DataLoader], List[DataLoader], List[Evaluator], str]:
    """Return dataloaders and evaluators."""
    data_root = "data/COCO"
    train_resolution = (800, 1333)
    test_resolution = (800, 1333)
    if is_training:
        train_loader = default_train_pipeline(
            COCO(data_root, split="train2017", data_backend=HDF5Backend()),
            batch_size,
            num_workers,
            train_resolution,
            with_mask=True,
        )
    else:
        train_loader = None
    test_loader = default_test_pipeline(
        COCO(data_root, split="val2017", data_backend=HDF5Backend()),
        1,
        1,
        test_resolution,
    )
    test_evals = [COCOEvaluator(data_root), COCOEvaluator(data_root, "segm")]
    test_metric = "COCO_AP"
    return train_loader, test_loader, test_evals, test_metric


def data_connector(mode: str, data: DictData):
    """Data connector."""
    if mode == "train":
        data_keys = {
            "images": "images",
            "input_hw": "images_hw",
            "boxes2d": "target_boxes",
            "boxes2d_classes": "target_classes",
        }
    elif mode == "loss":
        data_keys = {
            "input_hw": "images_hw",
            "boxes2d": "target_boxes",
            "masks": "target_masks",
        }
    else:
        data_keys = {
            "images": "images",
            "input_hw": "images_hw",
            "original_hw": "original_hw",
        }
    return {v: data[k] for k, v in data_keys.items()}


def train(num_gpus: int, ckpt: str) -> None:
    """Training."""
    # parameters
    log_step = 100
    num_iters = 40000
    batch_size = int(8 * (num_gpus / 8))
    learning_rate = 0.02 / 16 * batch_size
    device = torch.device("cuda")
    save_prefix = "vis4d-workspace/test/maskrcnn_coco_epoch"

    # data loaders and evaluators
    train_loader, test_loader, test_evals, test_metric = get_dataloaders(
        True, batch_size
    )
    assert train_loader is not None

    # model
    FCNResNet = FCNResNet(base_model=args.base_model, resize=(520, 520))
    FCNResNet.to(device)

    if num_gpus > 1:
        mask_rcnn = nn.DataParallel(
            mask_rcnn, device_ids=[device, torch.device("cuda:1")]
        )

    # optimization
    optimizer = optim.Adam(
        FCNResNet.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0001,
    )
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, num_iters, power=0.9
    )
    warmup = LinearLRWarmup(0.001, 500)

    # run training
    training_loop(
        train_loader,
        test_loader,
        test_evals,
        test_metric,
        FCNResNet,
        mask_rcnn_loss,
        data_connector,
        optimizer,
        scheduler,
        num_epochs,
        log_step,
        learning_rate,
        save_prefix,
        warmup,
    )


def test(ckpt: str) -> None:
    """Testing."""
    # parameters
    device = torch.device("cuda")

    # data loaders and evaluators
    _, test_loader, test_evals, test_metric = get_dataloaders()

    # model
    FCNResNet = FCNResNet(base_model=args.base_model, resize=(520, 520))
    FCNResNet.to(device)

    # run testing
    testing_loop(
        test_loader, test_evals, test_metric, FCNResNet, data_connector
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    parser.add_argument("-n", "--num_gpus", default=1, help="number of gpus")
    args = parser.parse_args()
    if args.ckpt is None:
        train(args.num_gpus, args.ckpt)
    else:
        test(args.ckpt)
