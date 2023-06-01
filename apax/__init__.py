import os
import warnings

from jax.config import config as jax_config

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
jax_config.update("jax_enable_x64", True)
