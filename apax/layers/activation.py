from typing import Callable
import inspect

import jax


def get_activation_fn(name: str) -> Callable[[float], float]:
    """Get the activation function"""
    if not hasattr(jax.nn, name):
        raise AttributeError(
            f"jax.nn has no attribute {name}. Is not a valid activation function. See https://docs.jax.dev/en/latest/jax.nn.html for options."
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
        and (p.default == inspect._empty)
    ]
    if len(required_positional) != 1:
        raise TypeError(
            f"jax.nn.{name} is not a valid readout activation: expected exactly one required positional argument, but needs {len(required_positional)}."
        )

    return activation_fn
