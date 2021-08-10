"""Tool for benchmarking VisT models."""

import argparse
import itertools
import logging
from typing import List

import psutil
import torch
import tqdm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetFromList
from detectron2.engine import SimpleTrainer, hooks, launch
from detectron2.solver import build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger
from fvcore.common.timer import Timer
from torch.nn.parallel import DistributedDataParallel

from vist.config import Config, default_argument_parser, parse_config
from vist.data import build_test_loader, build_train_loader
from vist.engine.utils import to_detectron2
from vist.model import build_model

logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)


def ram_msg() -> str:
    """Return current RAM usage as string."""
    vram = psutil.virtual_memory()
    return "RAM Usage: {:.2f}/{:.2f} GB".format(
        (vram.total - vram.available) / 1024 ** 3, vram.total / 1024 ** 3
    )


def benchmark_data(cfg: Config) -> None:
    """Benchmark speed of data pipeline."""
    det2cfg = to_detectron2(cfg)

    logger.info("After spawning %s", ram_msg())
    timer = Timer()
    dataloader = build_train_loader(cfg.dataloader, det2cfg)
    logger.info("Initialize loader using %s seconds.", timer.seconds())

    timer.reset()
    itr = iter(dataloader)
    for i in range(10):  # warmup
        next(itr)
        if i == 0:
            startup_time = timer.seconds()
    logger.info("Startup time: %s seconds", startup_time)
    timer = Timer()
    max_iter = 1000
    for _ in tqdm.trange(max_iter):
        next(itr)
    logger.info(
        "%s iters (%s images) in %s seconds.",
        max_iter,
        max_iter * det2cfg.SOLVER.IMS_PER_BATCH,
        timer.seconds(),
    )

    # test for a few more rounds
    for k in range(10):
        logger.info("Iteration %s %s", k, ram_msg())
        timer = Timer()
        max_iter = 1000
        for _ in tqdm.trange(max_iter):
            next(itr)
        logger.info(
            "%s iters (%s images) in %s seconds.",
            max_iter,
            max_iter * det2cfg.SOLVER.IMS_PER_BATCH,
            timer.seconds(),
        )


def benchmark_train(cfg: Config) -> None:
    """Benchmark speed of training pipeline."""

    setup_logger(distributed_rank=comm.get_rank())
    cfg.solver.base_lr = 0.00001  # Avoid NaN loss in benchmark due to high LR
    det2cfg = to_detectron2(cfg)

    model = build_model(cfg.model).to(torch.device(det2cfg.MODEL.DEVICE))
    logger.info("Model:\n%s", model)
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    optimizer = build_optimizer(det2cfg, model)
    checkpointer = DetectionCheckpointer(model, optimizer=optimizer)
    checkpointer.load(det2cfg.MODEL.WEIGHTS)

    cfg.dataloader.workers_per_gpu = 0
    data_loader = build_train_loader(cfg.dataloader, det2cfg)
    dummy_data = list(itertools.islice(data_loader, 100))

    def func():  # type: ignore
        data = DatasetFromList(dummy_data, copy=False, serialize=False)
        while True:
            yield from data

    max_iter = 400
    trainer = SimpleTrainer(model, func(), optimizer)  # type: ignore
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.PeriodicWriter([CommonMetricPrinter(max_iter)]),
            hooks.TorchProfiler(
                lambda trainer: trainer.iter == max_iter - 1,
                det2cfg.OUTPUT_DIR,
                save_tensorboard=True,
            ),
        ]
    )
    trainer.train(1, max_iter)


def benchmark_test(cfg: Config) -> None:
    """Benchmark speed of testing pipeline."""
    det2cfg = to_detectron2(cfg)

    model = build_model(cfg.model).to(torch.device(det2cfg.MODEL.DEVICE))
    model.eval()
    logger.info("Model:\n%s", model)
    DetectionCheckpointer(model).load(det2cfg.MODEL.WEIGHTS)

    cfg.dataloader.workers_per_gpu = 0
    data_loader = build_test_loader(
        cfg.dataloader,
        det2cfg,
        cfg.test[0].name,
        cfg.test[0].inference_sampling,
    )
    dummy_data = DatasetFromList(
        list(itertools.islice(data_loader, 100)), copy=False, serialize=False
    )

    def func():  # type: ignore
        while True:
            yield from dummy_data

    for k in range(5):  # warmup
        model(dummy_data[k])

    max_iter = 300
    timer = Timer()
    with tqdm.tqdm(total=max_iter) as pbar:
        for idx, d in enumerate(func()):  # type: ignore
            if idx == max_iter:
                break
            model(d)
            pbar.update()
    logger.info("%s iters in %s seconds.", max_iter, timer.seconds())


def _modify_choices(
    _parser: argparse.ArgumentParser, dest: str, choices: List[str]
) -> None:
    """Modify argparser argument."""
    for action in _parser._actions:  # pylint: disable=protected-access
        if action.dest == dest:
            action.choices = choices
            return

    raise AssertionError("argument {} not found".format(dest))


if __name__ == "__main__":
    parser = default_argument_parser()
    _modify_choices(parser, "action", ["train", "test", "data"])
    args = parser.parse_args()
    config = parse_config(args)

    logger.info("Environment info:\n%s", collect_env_info())
    if args.action == "data":
        f = benchmark_data
        logger.info("Initial %s", ram_msg())
    elif args.action == "train":
        # Note: training speed may not be representative.
        # The training cost of e.g. an R-CNN model varies with the content of
        # the data and the quality of the model.
        f = benchmark_train
    else:
        f = benchmark_test
        # only benchmark single-GPU inference.
        assert args.num_gpus == 1 and args.num_machines == 1

    launch(
        f,
        config.launch.num_gpus,
        config.launch.num_machines,
        config.launch.machine_rank,
        config.launch.dist_url,
        args=(config,),
    )
