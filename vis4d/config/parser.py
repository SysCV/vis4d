"""Parser for config files that can be used with absl flags."""
from __future__ import annotations

import logging
import sys
import traceback

import yaml
from absl import flags
from ml_collections import ConfigDict
from ml_collections.config_flags.config_flags import (
    _ConfigFlag,
    _ErrorConfig,
    _LoadConfigModule,
    _LockConfig,
)


class _ConfigFileParser(flags.ArgumentParser):  # type: ignore
    """Parser for config files."""

    def __init__(
        self,
        name: str,
        lock_config: bool = True,
        method_name: str = "get_config",
    ) -> None:
        self.name = name
        self._lock_config = lock_config
        self.method_name = method_name

    def parse(  # pylint: disable=arguments-renamed
        self, path: str
    ) -> ConfigDict | _ErrorConfig:
        """Loads a config module from `path` and returns the `method_name()`.

        This implementation is based on the original ml_collections and
        modified to allow for a custom method name.

        If a colon is present in `path`, everything to the right of the first
        colon is passed to `method_name` as an argument. This allows the
        structure of what
        is returned to be modified, which is useful when performing complex
        hyperparameter sweeps.

        Args:
          path: string, path pointing to the config file to execute. May also
              contain a config_string argument, e.g. be of the form
              "config.py:some_configuration".

        Returns:
          Result of calling `method_name` in the specified module.
        """
        # This will be a 2 element list iff extra configuration args are
        # present.
        split_path = path.split(":", 1)
        try:
            config_module = _LoadConfigModule(
                f"{self.name}_config", split_path[0]
            )
            config = getattr(config_module, self.method_name)(*split_path[1:])
            if config is None:
                logging.warning(
                    "%s:%s() returned None, did you forget a return "
                    "statement?",
                    path,
                    self.method_name,
                )
        except IOError as e:
            # Don't raise the error unless/until the config is
            # actually accessed.
            config = _ErrorConfig(e)
        # Third party flags library catches TypeError and ValueError
        # and rethrows,
        # removing useful information unless it is added here (b/63877430):
        except (TypeError, ValueError) as e:
            error_trace = traceback.format_exc()
            raise type(e)(
                "Error whilst parsing config file:\n\n" + error_trace
            )

        if self._lock_config:
            _LockConfig(config)

        return config

    def flag_type(self) -> str:
        return "config object"


class ConfigFileParser(_ConfigFileParser):
    """Parser for config files.

    Note, this wraps internal functions of the ml_collections code and might
    be fragile!
    """

    def parse(self, path: str) -> ConfigDict:
        """Returns the config object for a given path.

        If a colon is present in `path`, everything to the right of the first
        colon is passed to `get_config` as an argument. This allows the
        structure of what  is returned to be modified.

        Works with .py file that contain a get_config() function and .yaml.

        Args:
          path (string): path pointing to the config file to execute. May also
              contain a config_string argument, e.g. be of the form
              "config.py:some_configuration" or "config.yaml".
        Returns (ConfigDict):
          ConfigDict located at 'path'
        """
        if path.split(".")[-1] == "yaml":
            with open(path, "r", encoding="utf-8") as yaml_file:
                data_dict = ConfigDict(yaml.safe_load(yaml_file))

                if self._lock_config:
                    data_dict.lock()
                return data_dict
        else:
            return super().parse(path)

    def flag_type(self) -> str:
        """The flag type of this object.

        Returns:
            str: config object
        """
        return "config object"


def DEFINE_config_file(  # pylint: disable=invalid-name
    name: str,
    default: str | None = None,
    help_string: str = "path to config file [.py |.yaml].",
    lock_config: bool = True,
    method_name: str = "get_config",
) -> flags.FlagHolder:
    """Registers a new flag for a config file.

    Args:
        name (str): The name of the flag (e.g. config for --config flag)
        default (str | None, optional): Default Value. Defaults to None.
        help_string (str, optional): Help String.
            Defaults to "path to config file.".
        lock_config (bool, optional): Whether or note to lock the returned
            config. Defaults to True.
        method_name (str, optional): Name of the method to call in the config.

    Returns:
        flags.FlagHolder: Flag holder instance.
    """
    parser = ConfigFileParser(
        name=name, lock_config=lock_config, method_name=method_name
    )
    flag = _ConfigFlag(
        parser=parser,
        serializer=flags.ArgumentSerializer(),
        name=name,
        default=default,
        help_string=help_string,
        flag_values=flags.FLAGS,
    )

    # Get the module name for the frame at depth 1 in the call stack.
    module_name = sys._getframe(  # pylint: disable=protected-access
        1
    ).f_globals.get("__name__", None)
    module_name = sys.argv[0] if module_name == "__main__" else module_name
    return flags.DEFINE_flag(flag, flags.FLAGS, module_name=module_name)
