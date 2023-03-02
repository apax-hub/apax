import warnings

import tensorflow as tf
from jax.config import config as jax_config

tf.config.set_visible_devices([], "GPU")

warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
jax_config.update("jax_enable_x64", True)
