"""Default argument parser for vist."""
import argparse

from .config import Launch


def default_argument_parser() -> argparse.ArgumentParser:
    """Create a parser with common vist arguments."""
    schema = Launch.schema()
    parser = argparse.ArgumentParser(description=schema["description"])
    parser.add_argument(
        "action",
        type=str,
        choices=["train", "test", "predict"],
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

    for key, val in schema["properties"].items():
        if Launch.__fields__[key].type_ == bool:
            if not val["default"]:
                parser.add_argument("--" + key, action="store_true")
            else:
                parser.add_argument(
                    "--" + key, action="store_false"
                )  # pragma: no cover
        else:
            parser.add_argument(
                "--" + key,
                default=None,
                type=Launch.__fields__[key].type_,
            )
    return parser
