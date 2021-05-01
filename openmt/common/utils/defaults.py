"""Default argument parser for openmt."""
import argparse

from openmt.config import Launch


def default_argument_parser() -> argparse.ArgumentParser:
    """Create a parser with common openmt arguments."""
    schema = Launch.schema()
    parser = argparse.ArgumentParser(description=schema["description"])
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
        "--cfg-options",
        default="",
        help="additional config parameters in format key=value separated by "
        "commas, e.g. dataloader.workers_per_gpu=1,solver.base_lr=0.01",
    )

    for key in schema["properties"]:
        parser.add_argument(
            "--" + key,
            default=None,
            type=Launch.__fields__[key].type_,
        )
    return parser
