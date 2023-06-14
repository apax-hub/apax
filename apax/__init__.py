import os

from jax.config import config as jax_config

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax_config.update("jax_enable_x64", True)
