import os
import warnings

import jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore", message=".*os.fork()*")

from apax._version import __version__, __version_tuple__  # noqa: F401
