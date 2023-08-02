import logging
import os
from typing import Union

import yaml

from apax.config.md_config import MDConfig
from apax.config.train_config import Config

log = logging.getLogger(__name__)


def parse_config(config: Union[str, os.PathLike, dict], mode: str = "train") -> Config:
    """Load the training configuration from file or a dictionary.

    Attributes
    ----------
        config: Path to the config file or a dictionary
        containing the config.
    """
    log.info("Loading user config")
    if isinstance(config, (str, os.PathLike)):
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

    if mode == "train":
        config = Config.parse_obj(config)
    elif mode == "md":
        config = MDConfig.parse_obj(config)

    return config
