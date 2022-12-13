from typing import Any, Sequence

import haiku as hk
import jax
import jax.numpy as jnp


class NTKWeights(hk.initializers.Initializer):
    """NTK Initialization"""

    def __init__(
        self,
        raw_weight_factor: float = 1.0,
    ):
        """Constructs a :class:`NTKWeights` initializer.
        Args:
        """
        # TODO NEEDS TO BE CHECKED FOR CORRECTNESS
        self.raw_weight_factor = raw_weight_factor

    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        weight_factor = self.raw_weight_factor * jnp.sqrt(1.0 / shape[0])
        param = weight_factor * jax.random.normal(hk.next_rng_key(), shape, dtype)
        return param


class NTKBias(hk.initializers.Initializer):
    """NTK Initialization"""

    def __init__(
        self,
        bias_factor: float = 0.1,
    ):
        """Constructs a :class:`NTKBias` initializer.
        Args:
        """
        # TODO NEEDS TO BE CHECKED FOR CORRECTNESS
        self.bias_factor = bias_factor

    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        param = self.bias_factor * jnp.zeros(shape, dtype)
        return param
