import warnings

import tensorflow as tf
from jax.config import config as jax_config

# Disable all GPUS

# tf.config.experimental.set_visible_devices([], "GPU")


tf.config.set_visible_devices([], "GPU")
# visible_devices = tf.config.get_visible_devices()
# for device in visible_devices:
#     assert device.device_type != 'GPU'

warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
jax_config.update("jax_enable_x64", True)
