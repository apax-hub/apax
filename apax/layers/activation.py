import jax
from jax import Array


def swish(x: Array) -> Array:
    out = 1.6765324703310907 * jax.nn.swish(x)
    return out
