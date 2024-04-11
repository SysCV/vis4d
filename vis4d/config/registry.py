"""Utility function for registering config files."""

from __future__ import annotations

import glob
import os
import pathlib
import warnings
from typing import Callable, Union

import yaml
from ml_collections import ConfigDict
from ml_collections.config_flags.config_flags import _LoadConfigModule

from vis4d.common.dict import flatten_dict, get_dict_nested
from vis4d.common.typing import ArgsType
from vis4d.common.util import create_did_you_mean_msg
from vis4d.config.config_dict import FieldConfigDict
from vis4d.zoo import AVAILABLE_MODELS

MODEL_ZOO_FOLDER = str(
    (pathlib.Path(os.path.dirname(__file__)) / ".." / "zoo").resolve()
)

# Paths that are used to search for config files.
REGISTERED_CONFIG_PATHS = [MODEL_ZOO_FOLDER]


TFunc = Callable[[ArgsType], ArgsType]
TfuncConfDict = Union[Callable[[ArgsType], ConfigDict], type]


def register_config(
    category: str, name: str
) -> Callable[[TfuncConfDict], None]:
    """Register a config in the model zoo for the given name and category.

    The config will then be available via `get_config_by_name` utilities and
    located in the AVAILABLE_MODELS dictionary located at
    [category][name].

    Args:
        category: Category of the config.
        name: Name of the config.

    Returns:
        The decorator.
    """

    def decorator(fnc_or_clazz: TfuncConfDict) -> None:
        """Decorator for registering a config.

        Args:
            fnc_or_clazz: Function or class to register. If a function is
                passed, it will be wrapped in a class and the class will be
                registered. If a class is passed, it will be registered
                directly.
        """
        if callable(fnc_or_clazz):
            # Directly annotated get_config function. Wrap it and register it.
            class Wrapper:
                """Wrapper class."""

                def get_config(
                    self, *args: ArgsType, **kwargs: ArgsType
                ) -> ConfigDict:
                    """Resolves the get_config function."""
                    return fnc_or_clazz(*args, **kwargs)

            module = Wrapper()
        else:
            # Directly annotated class. Register it.
            module = fnc_or_clazz

        # Register the config
        if category not in AVAILABLE_MODELS:
            AVAILABLE_MODELS[category] = {}

        assert isinstance(AVAILABLE_MODELS[category], dict)

        AVAILABLE_MODELS[category][name] = module

    return decorator


def _resolve_config_path(path: str) -> str:
    """Resolve the path of a config file.

    Args:
        path: Name or path of the config.
            If the config is not found at this location,
            the function will look for the config in the model zoo folder.

    Returns:
        The resolved path of the config.

    Raises:
        ValueError: If the config is not found.
    """
    if os.path.exists(path):
        return path

    # Check for duplicate paths.
    found_paths: list[str] = []
    all_paths = []

    for p in REGISTERED_CONFIG_PATHS:
        paths = sorted(
            glob.glob(
                os.path.join(p, f"**/*{ os.path.splitext(path)[-1]}"),
                recursive=True,
            )
        )
        print(
            paths,
            "lookup",
            os.path.join(p, f"**/*{ os.path.splitext(path)[-1]}"),
        )
        for cfg_path in paths:
            if cfg_path.endswith(path):
                found_paths.append(cfg_path)
        all_paths.extend(paths)

    if len(found_paths) > 1:
        warnings.warn(
            f"Found multiple paths for config {path}:"
            f"{found_paths}. Will load the config from the first  one!"
        )
    elif len(found_paths) == 0:
        hint = create_did_you_mean_msg(
            [*all_paths, *[os.path.basename(p) for p in all_paths]], path
        )
        raise ValueError(
            f"Could not find config {path}. \n"
            f"The file does not exists at the path {path} or "
            f"in the dedicated locations at {REGISTERED_CONFIG_PATHS}. \n"
            f"Please check the path or add the config to the model zoo. \n"
            f"Current working directory: {os.getcwd()}\n {hint}"
        )
    return found_paths[0]


def _load_yaml_config(name_or_path: str) -> FieldConfigDict:
    """Loads a .yaml configuration file.

    Args:
        name_or_path: Name or path of the config.
            If the config is not found at this location, $
            the function will look for the config in the model zoo folder.

    Returns:
        The config for the experiment.
    """
    path = _resolve_config_path(name_or_path)
    with open(path, "r", encoding="utf-8") as yaml_file:
        return FieldConfigDict(yaml.load(yaml_file, Loader=yaml.UnsafeLoader))


def _load_py_config(
    name_or_path: str, *args: ArgsType, method_name: str = "get_config"
) -> ConfigDict:
    """Loads a .py configuration file.

    Args:
        name_or_path: Name or path of the config.
            If the config is not found at this location,
            the function will look for the config in the model zoo folder.
        *args: Additional arguments to pass to the config.
        method_name: Name of the method to call from the file to get the
            config. Defaults to "get_config".

    Returns:
        The config for the experiment.
    """
    path = _resolve_config_path(name_or_path)
    config_module = _LoadConfigModule(f"{os.path.basename(path)}_config", path)
    cfg = getattr(config_module, method_name)(*args)
    assert isinstance(cfg, ConfigDict)
    return cfg


def _get_registered_configs(
    config_name: str, *args: ArgsType, method_name: str = "get_config"
) -> ConfigDict:
    """Get a model from the registered config locations.

    Args:
        config_name: Name of the config. This can either be
            the full path of the config relative to the registered locations
            or the name of the config.
            If the config matches multiple configs (e.g. if there are two
            conflicting config a/cfg and b/cfg) or if it is not found,
            a ValueError is raised.
        *args: Additional arguments to pass to the config.
        method_name: Name of the method to call from the file to get the
            config. Defaults to "get_config".

    Raises:
        ValueError: If the config is not found.

    Returns:
        The Config.
    """
    models = flatten_dict(AVAILABLE_MODELS, os.path.sep)
    # check if there is an absolute match for the config
    if config_name in models:
        module = get_dict_nested(
            AVAILABLE_MODELS, config_name.split(os.path.sep)
        )
        return getattr(module, method_name)(*args)
    # check if there is a partial match for the config
    matches = {}
    for model in models:
        if model.endswith(config_name):
            matches[model] = get_dict_nested(
                AVAILABLE_MODELS, model.split(os.path.sep)
            )

    if len(matches) > 1:
        raise ValueError(
            f"Found multiple configs matching {config_name}:"
            f"{matches.keys()}.\nPlease specify a unique config name."
        )
    if len(matches) == 0:
        msg = create_did_you_mean_msg(
            [*models, *[os.path.basename(m) for m in models]], config_name
        )
        raise ValueError(msg)

    module = list(matches.values())[0]
    return getattr(module, method_name)(*args)


def get_config_by_name(
    name_or_path: str, *args: ArgsType, method_name: str = "get_config"
) -> ConfigDict:
    """Get a config by name or path.

    Args:
        name_or_path: Name or path of the config.
            If the path has a .yaml or .py extension, the function will
            load the config from the file.
            Otherwise, the function will try to resolve the config from the
            registered config locations. You can specify a config by its full
            registered path (e.g. "a/b/cfg") or by its name (e.g. "cfg").
        *args: Additional arguments to pass to the config.
        method_name: Name of the method to call from the file to get the
            config. Defaults to "get_config".

    Returns:
        The config.

    Raises:
        ValueError: If the config is not found.
    """
    if name_or_path.endswith(".yaml"):
        return _load_yaml_config(name_or_path)
    if name_or_path.endswith(".py"):
        return _load_py_config(name_or_path, *args, method_name=method_name)
    return _get_registered_configs(
        name_or_path, *args, method_name=method_name
    )
