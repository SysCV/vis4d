"""Mask RCNN COCO training example."""
import argparse
import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from vis4d.data.datasets.coco import COCO
from vis4d.data.io import HDF5Backend
from vis4d.eval import COCOEvaluator, Evaluator
from vis4d.model.detect.mask_rcnn import MaskRCNN, MaskRCNNLoss
from vis4d.optim.warmup import LinearLRWarmup
from vis4d.run.data.detect import default_test_pipeline, default_train_pipeline
from vis4d.run.test import testing_loop
from vis4d.run.train import training_loop

warnings.filterwarnings("ignore")


def get_dataloaders(
    is_training: bool = False, batch_size: int = 1
) -> Tuple[Optional[DataLoader], List[DataLoader], List[Evaluator], str]:
    """Return dataloaders and evaluators."""
    data_root = "data/COCO"
    train_resolution = (800, 1333)
    test_resolution = (800, 1333)
    if is_training:
        train_loader = default_train_pipeline(
            COCO(data_root, split="train2017", data_backend=HDF5Backend()),
            batch_size,
            train_resolution,
            with_mask=True,
        )
    else:
        train_loader = None
    test_loader = default_test_pipeline(
        COCO(data_root, split="val2017", data_backend=HDF5Backend()),
        1,
        test_resolution,
    )
    test_evals = [COCOEvaluator(data_root), COCOEvaluator(data_root, "segm")]
    test_metric = "COCO_AP"
    return train_loader, test_loader, test_evals, test_metric


def train(args: argparse.Namespace) -> None:
    """Training."""
    # parameters
    log_step = 100
    num_epochs = 12
    batch_size = int(8 * (args.num_gpus / 8))
    learning_rate = 0.02 / 16 * batch_size
    device = torch.device("cuda")
    save_prefix = "vis4d-workspace/test/maskrcnn_coco_epoch"

    # data loaders and evaluators
    train_loader, test_loader, test_evals, test_metric = get_dataloaders(
        True, batch_size
    )
    assert train_loader is not None

    # model
    mask_rcnn = MaskRCNN(num_classes=80)
    mask_rcnn.to(device)
    mask_rcnn_loss = MaskRCNNLoss(
        mask_rcnn.anchor_gen,
        mask_rcnn.rpn_bbox_encoder,
        mask_rcnn.rcnn_bbox_encoder,
    )
    if args.num_gpus > 1:
        mask_rcnn = nn.DataParallel(
            mask_rcnn, device_ids=[device, torch.device("cuda:1")]
        )
    model_train_keys = ["images", "input_hw", "boxes2d", "boxes2d_classes"]
    model_test_keys = ["images", "input_hw", "original_hw"]
    loss_keys = ["input_hw", "boxes2d", "masks"]

    # optimization
    optimizer = optim.SGD(
        mask_rcnn.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0001,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[8, 11], gamma=0.1
    )
    warmup = LinearLRWarmup(0.001, 500)

    # run training
    training_loop(
        train_loader,
        test_loader,
        test_evals,
        test_metric,
        mask_rcnn,
        mask_rcnn_loss,
        model_train_keys,
        model_test_keys,
        loss_keys,
        optimizer,
        scheduler,
        num_epochs,
        log_step,
        learning_rate,
        save_prefix,
        warmup,
    )


def test(args: argparse.Namespace) -> None:
    """Testing."""
    # parameters
    device = torch.device("cuda")

    # data loaders and evaluators
    _, test_loader, test_evals, test_metric = get_dataloaders()

    # model
    mask_rcnn = MaskRCNN(num_classes=80, weights=args.ckpt)
    mask_rcnn.to(device)
    model_test_keys = ["images", "input_hw", "original_hw"]

    # run testing
    testing_loop(
        test_loader, test_evals, test_metric, mask_rcnn, model_test_keys
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    parser.add_argument("-n", "--num_gpus", default=1, help="number of gpus")
    args = parser.parse_args()
    if args.ckpt is None:
        train(args)
    else:
        test(args)