import logging
import os
from collections.abc import MutableMapping
from typing import Any, Union

import yaml

from apax.config.md_config import MDConfig
from apax.config.optuna_config import OptunaConfig
from apax.config.train_config import Config

log = logging.getLogger(__name__)


def parse_config(config: Union[str, os.PathLike, dict], mode: str = "train") -> Config:
    """Load the training configuration from file or a dictionary.

    Parameters
    ----------
        config : str | os.PathLike | dict
            Path to the config file or a dictionary
            containing the config.
        mode: str, default = train
            Defines if the config is validated for training ("train"),
            MD simulation ("md") or hyperparameter optimization ("optuna")
    """
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

    if mode == "train":
        config = Config.model_validate(config)
    elif mode == "md":
        config = MDConfig.model_validate(config)
    elif mode == "optuna":
        config = OptunaConfig.model_validate(config)

    return config


def flatten(dictionary, parent_key="", separator="_"):
    """https://stackoverflow.com/questions/6027558/
    flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _unflatten_rec(
    flat_dct: dict[str, Any],
    nested_keys: dict[str, tuple[None | dict[str, ...]]],
    parent_key: str = "",
    separator: str = "_",
    dct: dict | None = None,
    seen_params: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if dct is None:
        dct = {}
    if seen_params is None:
        seen_params = []
    for param, value in flat_dct.items():
        for key, nested_key in nested_keys.items():
            if not param.startswith(key) and not param.startswith(
                separator.join((parent_key, key))
            ):
                continue
            if key not in dct:
                dct[key] = {}
            for val in nested_key:
                if isinstance(val, MutableMapping):
                    dct[key], new_seen_params = _unflatten_rec(
                        flat_dct, val, dct=dct[key], parent_key=key
                    )
                    seen_params.extend(new_seen_params)
                else:
                    if param in seen_params:
                        continue

                    new_param = param.lstrip(f"{parent_key}{separator}")
                    new_param = new_param.lstrip(f"{key}{separator}")
                    dct[key][new_param] = value
                    seen_params.append(param)
    return dct, seen_params


def unflatten(
    flat_dct: dict[str, Any],
    nested_keys: dict[str, tuple[None | dict[str, ...]]],
    separator: str = "_",
) -> dict[str, Any]:
    unflattened_dct, seen_params = _unflatten_rec(
        flat_dct, nested_keys, separator=separator
    )
    # If it should be in the uppermost level, add it afterwards
    for param, value in flat_dct.items():
        if param not in seen_params:
            unflattened_dct[param] = value
    return unflattened_dct
