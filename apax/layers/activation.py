import jax


def swish(x):
    out = 1.6765324703310907 * jax.nn.swish(x)
    return out
