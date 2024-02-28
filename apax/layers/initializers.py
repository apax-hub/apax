from typing import Any

import jax.numpy as jnp
from jax import random
from jax._src import dtypes
from jax.nn.initializers import Initializer

Array = Any
KeyArray = Array

DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any


def uniform_range(minval, maxval, dtype: DTypeLikeInexact = jnp.float_) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays
    in a specified range.
    """

    def init(key: KeyArray, shape, dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.uniform(key, shape, dtype, minval, maxval)

    return init
