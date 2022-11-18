"""Faster RCNN COCO training example."""
import argparse
import os
import warnings
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from vis4d.common.distributed import get_world_size
from vis4d.data import DictData
from vis4d.data.datasets.coco import COCO
from vis4d.data.io import HDF5Backend
from vis4d.eval import COCOEvaluator, Evaluator
from vis4d.model.detect.faster_rcnn import FasterRCNN, FasterRCNNLoss
from vis4d.optim.warmup import LinearLRWarmup
from vis4d.run.data.detect import default_test_pipeline, default_train_pipeline
from vis4d.run.test import testing_loop
from vis4d.run.train import training_loop

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
        )
    else:
        train_loader = None
    test_loader = default_test_pipeline(
        COCO(data_root, split="val2017", data_backend=HDF5Backend()),
        1,
        1,
        test_resolution,
    )
    test_evals = [COCOEvaluator(data_root)]
    test_metric = "COCO_AP"
    return train_loader, test_loader, test_evals, test_metric


def data_connector(mode: str, data: DictData):
    """Data connector."""
    if mode == "train":
        data_keys = {
            "images": "images",
            "input_hw": "input_hw",
            "boxes2d": "boxes2d",
            "boxes2d_classes": "boxes2d_classes",
        }
    elif mode == "loss":
        data_keys = {"input_hw": "input_hw", "boxes2d": "boxes2d"}
    else:
        data_keys = {
            "images": "images",
            "input_hw": "input_hw",
            "original_hw": "original_hw",
        }
    return {v: data[k] for k, v in data_keys.items()}


def train(ckpt: str, gpu_id: int = 0) -> None:
    """Training."""
    # parameters
    log_step = 100
    num_epochs = 12
    batch_size = int(16 * (get_world_size() / 8))
    learning_rate = 0.02 / 16 * batch_size
    device = torch.device(f"cuda:{gpu_id}")
    save_prefix = "vis4d-workspace/test/frcnn_coco_epoch"

    # data loaders and evaluators
    train_loader, test_loader, test_evals, test_metric = get_dataloaders(
        True,
        batch_size,
    )
    assert train_loader is not None
    torch.backends.cudnn.benchmark = True

    # model
    faster_rcnn = FasterRCNN(num_classes=80, weights=ckpt)
    faster_rcnn.to(device)
    faster_rcnn_loss = FasterRCNNLoss()
    if get_world_size() > 1:
        faster_rcnn = DDP(faster_rcnn, device_ids=[gpu_id])

    # optimization
    optimizer = optim.SGD(
        faster_rcnn.parameters(),
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
        faster_rcnn,
        faster_rcnn_loss,
        data_connector,
        optimizer,
        scheduler,
        num_epochs,
        log_step,
        learning_rate,
        save_prefix,
        warmup,
    )


def test(ckpt: str, gpu_id: int = 0) -> None:
    """Testing."""
    # parameters
    device = torch.device(f"cuda:{gpu_id}")

    # data loaders and evaluators
    _, test_loader, test_evals, test_metric = get_dataloaders()

    # model
    faster_rcnn = FasterRCNN(num_classes=80, weights=ckpt)
    faster_rcnn.to(device)
    if get_world_size() > 1:
        faster_rcnn = DDP(faster_rcnn, device_ids=[gpu_id])

    # run testing
    testing_loop(
        test_loader, test_evals, test_metric, faster_rcnn, data_connector
    )


def ddp_setup(rank: int, world_size: int) -> None:
    """Setup DDP environment and init processes.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank: int, ckpt: str, world_size: int) -> None:
    """Main script setting up DDP, executing action, terminating."""
    ddp_setup(rank, world_size)
    if ckpt is None:
        train(ckpt, rank)
    else:
        test(ckpt, rank)  # TODO test testing loop with DDP
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO train/eval.")
    parser.add_argument(
        "-c", "--ckpt", default=None, help="path of model to eval"
    )
    parser.add_argument(
        "-n", "--num_gpus", type=int, default=1, help="number of gpus"
    )
    args = parser.parse_args()
    mp.spawn(
        main,
        args=(
            args.ckpt,
            args.num_gpus,
        ),
        nprocs=args.num_gpus,
    )
