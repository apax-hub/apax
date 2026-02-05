import logging
import os
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from apax.config.md_config import MDConfig
from apax.config.optuna_config import OptunaConfig
from apax.config.train_config import Config

log = logging.getLogger(__name__)


def parse_config(
    config: Union[str, os.PathLike, Dict], mode: str = "train"
) -> Union[Config, MDConfig, OptunaConfig]:
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


def flatten(dictionary: Dict, parent_key: str = "", separator: str = "_") -> Dict:
    """https://stackoverflow.com/questions/6027558/
    flatten-nested-dictionaries-compressing-keys
    """
    items: List[Tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _unflatten_rec(
    flat_dct: Dict[str, Any],
    nested_keys: Dict[str, Tuple[Union[None, Dict[str, Any]], ...]],
    parent_key: str = "",
    separator: str = "_",
    dct: Optional[Dict] = None,
    seen_params: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    if dct is None:
        dct = {}
    if seen_params is None:
        seen_params = []
    for param, value in flat_dct.items():
        for key, nested_key_tuple in nested_keys.items():
            # Check if param starts with the current key or its full path including parent_key
            current_param_prefix = (
                f"{parent_key}{separator}{key}" if parent_key else key
            )
            if not param.startswith(current_param_prefix):
                continue

            if key not in dct:
                dct[key] = {}

            # Iterate over the tuple, assuming there's only one nested_key dict if any
            for nested_key in nested_key_tuple:
                if isinstance(nested_key, MutableMapping):
                    dct[key], new_seen_params = _unflatten_rec(
                        flat_dct,
                        nested_key,
                        dct=dct[key],
                        parent_key=current_param_prefix,
                        separator=separator,
                        seen_params=seen_params,
                    )
                    # Extend seen_params directly in the recursive call, no need to return from recursion explicitly
                else:  # if nested_key is None or other non-mapping type
                    if param in seen_params:
                        continue

                    new_param = param.lstrip(f"{current_param_prefix}{separator}")
                    # Ensure the new_param is not empty after stripping
                    if new_param:
                        dct[key][new_param] = value
                        seen_params.append(param)
    return dct, seen_params


def unflatten(
    flat_dct: Dict[str, Any],
    nested_keys: Dict[str, Tuple[Union[None, Dict[str, Any]], ...]],
    separator: str = "_",
) -> Dict[str, Any]:
    unflattened_dct, seen_params = _unflatten_rec(
        flat_dct, nested_keys, separator=separator
    )
    # If it should be in the uppermost level, add it afterwards
    for param, value in flat_dct.items():
        if param not in seen_params:
            unflattened_dct[param] = value
    return unflattened_dct
