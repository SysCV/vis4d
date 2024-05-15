"""CLI interface."""

from __future__ import annotations

from absl import app  # pylint: disable=no-name-in-module

from vis4d.common import ArgsType
from vis4d.zoo import AVAILABLE_MODELS


def main(argv: ArgsType) -> None:
    """Main entry point for the model zoo."""
    assert len(argv) > 1, "Command must be specified: `list`"
    if argv[1] == "list":
        for ds, models in AVAILABLE_MODELS.items():
            print(ds)
            model_names = list(models.keys())
            for model in model_names[:-1]:
                print(" ├─", model)
            print(" └─", model_names[-1])
    else:
        raise ValueError(f"Invalid command. {argv[1]}")


def entrypoint() -> None:
    """Entry point for the CLI."""
    app.run(main)
