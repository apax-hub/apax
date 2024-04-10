import logging
import os
from collections.abc import MutableMapping
from typing import Union

import yaml

from apax.config.md_config import MDConfig
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
            Defines if the config is validated for training ("train")
            or MD simulation("md").
    """
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

    if mode == "train":
        config = Config.model_validate(config)
    elif mode == "md":
        config = MDConfig.model_validate(config)

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
