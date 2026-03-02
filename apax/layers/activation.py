from typing import Callable

import jax


def get_activation_fn(name: str) -> Callable[[float], float]:
    """Get the activation function"""
    if not hasattr(jax.nn, name):
        raise AttributeError(
            f"jax.nn has no attribute {name}. Is not a proper activation function"
        )

    activation_fn = getattr(jax.nn, name)

    if not callable(activation_fn):
        raise TypeError(f"jax.nn.{name} is not callable")

    return activation_fn
