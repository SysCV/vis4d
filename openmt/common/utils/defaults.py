"""Default boilerplate logic for openmt."""
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

    for key in schema["properties"]:
        parser.add_argument(
            "--" + key,
            default=None,
            type=Launch.__fields__[key].type_,
        )
    return parser
