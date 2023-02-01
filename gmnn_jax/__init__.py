import warnings

from jax.config import config as jax_config
import tensorflow as tf


warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
tf.config.experimental.set_visible_devices([], "GPU")
jax_config.update("jax_enable_x64", True)
