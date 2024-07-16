from dataclasses import field
from typing import Any, Callable, List

import flax.linen as nn
import jax.numpy as jnp

from apax.layers.activation import swish
from apax.layers.ntk_linear import NTKLinear


class AtomisticReadout(nn.Module):
    units: List[int] = field(default_factory=lambda: [512, 512])
    activation_fn: Callable = swish
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = True
    n_shallow_ensemble: int = 0
    dtype: Any = jnp.float32

    def setup(self):
        readout_unit = [1]
        if self.n_shallow_ensemble > 0:
            readout_unit = [self.n_shallow_ensemble]
        units = list(self.units) + readout_unit
        dense = []
        for ii, n_hidden in enumerate(units):
            layer = NTKLinear(
                n_hidden,
                w_init=self.w_init,
                b_init=self.b_init,
                use_ntk=self.use_ntk,
                dtype=self.dtype,
                name=f"dense_{ii}",
            )
            dense.append(layer)
            if ii < len(units) - 1:
                dense.append(swish)
        self.sequential = nn.Sequential(dense, name="readout")

    def __call__(self, x):
        h = self.sequential(x)
        return h
