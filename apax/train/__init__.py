import warnings

import tensorflow as tf
from jax.config import config as jax_config

tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
jax_config.update("jax_enable_x64", True)


from apax.train import checkpoints, eval, loss, metrics
from apax.train.run import run
from apax.train.trainer import fit

__all__ = ["checkpoints", "loss", "metrics", "fit", "run", "eval"]
