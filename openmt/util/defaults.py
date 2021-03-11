"""Default boilerplate logic for openmt."""
import argparse


def default_argument_parser() -> argparse.ArgumentParser:
    """Create a parser with common openmt arguments."""
    parser = argparse.ArgumentParser(description="openmt options")
    parser.add_argument(
        "action",
        type=str,
        choices=["train", "predict"],
        help="Action to execute",
    )
    parser.add_argument(
        "--config", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory.",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        help="the rank of this machine (unique per machine)",
    )
    parser.add_argument(
        "--dist-url",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    return parser
