import warnings

import tensorflow as tf
from jax.config import config as jax_config

warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
tf.config.experimental.set_visible_devices([], "GPU")
jax_config.update("jax_enable_x64", True)
