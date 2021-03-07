"""Detection code."""

from detectron2.engine import launch
from .train import train
from .predict import predict

__all__ = ['train', 'predict']


def launch_module(module, args, cfg):
    """Launcher for detect modules."""

    launch(
        module,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, cfg),
    )
