import os

import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_enable_x64", True)
from apax.utils.helpers import setup_ase

setup_ase()
