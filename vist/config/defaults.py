"""Default argument parser for vist."""
import argparse

import pytorch_lightning as pl

from .config import Launch


def help_from_docstring(key: str, docstring: str) -> str:
    """Extract documentation for parameter key from class docstring."""
    result = docstring.split(key + ": ")[-1]
    result = result.split(": ", maxsplit=1)[0].rsplit(" ", 1)[0]
    return result


def default_argument_parser() -> argparse.ArgumentParser:
    """Create a parser with common vist arguments."""
    parser = argparse.ArgumentParser(description="VisT command line tool.")
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

    for key, val in Launch.schema()["properties"].items():
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
                help=help_from_docstring(key, Launch.__doc__),
            )

    pl.Trainer.add_argparse_args(parser)
    return parser
