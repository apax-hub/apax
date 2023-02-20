from dataclasses import field
from typing import Any, Callable, List

import flax.linen as nn
import jax.numpy as jnp

from gmnn_jax.layers.activation import swish
from gmnn_jax.layers.ntk_linear import NTKLinearFlax


class AtomisticReadout(nn.Module):
    units: List[int] = field(default_factory=lambda: [512, 512])
    activation_fn: Callable = swish
    b_init: str = "normal"
    dtype: Any = jnp.float32

    def setup(self):
        units = [u for u in self.units] + [1]
        dense = []
        for ii, n_hidden in enumerate(units):
            dense.append(
                NTKLinearFlax(
                    n_hidden, b_init=self.b_init, dtype=self.dtype, name=f"dense_{ii}"
                )
            )
            if ii < len(units) - 1:
                dense.append(swish)
        self.sequential = nn.Sequential(dense, name="readout")

    def __call__(self, x):
        h = self.sequential(x)
        return h
