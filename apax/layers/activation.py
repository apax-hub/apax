import inspect
from typing import Callable

import jax


def variance_preserving_swish(x) -> float:
    out = 1.6765324703310907 * jax.nn.swish(x)
    return out


def get_activation_fn(name: str) -> Callable[[float], float]:
    """Get the activation function from jax.nn. Also performs a bunch of checks
    to make sure that it is a valid activation function.

    Args:
        name (str): name of the activation function in jax.nn.

    Returns:
        activation_fn (Callable): Activation function, that takes in a float
            and returns a float.
    """

    if name == "variance_preserving_swish":
        # Keep backwards compatibility
        return variance_preserving_swish

    if not hasattr(jax.nn, name):
        raise AttributeError(
            f"jax.nn has no attribute {name}, see https://docs.jax.dev/en/latest/jax.nn.html for options."
        )

    activation_fn = getattr(jax.nn, name)

    if not callable(activation_fn):
        raise TypeError(f"jax.nn.{name} is not callable")

    signature = inspect.signature(activation_fn)
    required_positional = [
        p
        for p in signature.parameters.values()
        if (
            p.kind
            in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ]
        )
        and (p.default is inspect.Parameter.empty)
    ]
    if len(required_positional) != 1:
        raise TypeError(
            f"jax.nn.{name} is not a valid readout activation: expected exactly one required positional argument, but needs {len(required_positional)}."
        )

    return activation_fn
