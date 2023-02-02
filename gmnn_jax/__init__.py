import warnings

from jax.config import config as jax_config

warnings.filterwarnings(action="ignore", category=FutureWarning, module=r"jax.*scatter")
jax_config.update("jax_enable_x64", True)
