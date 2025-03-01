from dataclasses import field
from typing import Any, Callable, List

import flax.linen as nn
import jax.numpy as jnp

from apax.layers.activation import swish
from apax.layers.linear import Linear
from apax.utils.convert import str_to_dtype


class AtomisticReadout(nn.Module):
    units: List[int] = field(default_factory=lambda: [32, 32])
    activation_fn: Callable = swish
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = True
    n_shallow_ensemble: int = 0
    is_feature_fn: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        units = list(self.units)
        if not self.is_feature_fn:
            readout_unit = [1]
            if self.n_shallow_ensemble > 0:
                readout_unit = [self.n_shallow_ensemble]
            units += readout_unit

        dtype = str_to_dtype(self.dtype)

        dense = []
        for ii, n_hidden in enumerate(units):
            layer = Linear(
                n_hidden,
                w_init=self.w_init,
                b_init=self.b_init,
                use_ntk=self.use_ntk,
                dtype=dtype,
                name=f"dense_{ii}",
            )
            dense.append(layer)
            if ii < len(units) - 1:
                dense.append(swish)
        self.sequential = nn.Sequential(dense, name="readout")

    def __call__(self, x):
        h = self.sequential(x)
        # TODO should we move aggregation here?
        return h


class MultifidelityAtomisticReadout(nn.Module):
    units: List[int] = field(default_factory=lambda: [32, 32])
    num_fidelities: int = 1
    activation_fn: Callable = swish
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = False
    n_shallow_ensemble: int = 1
    is_feature_fn: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        units = list(self.units)
        if not self.is_feature_fn:
            readout_unit = [self.num_fidelities * self.n_shallow_ensemble]
            units += readout_unit

        dtype = str_to_dtype(self.dtype)
        # sequentials = []
        # for i in range(self.num_fidelities):
        #     print("i",i)
        #     seq = self.make_sequential(units, dtype, i)
        #     print(seq)
        #     sequentials.append(seq)
        # self.sequentials = sequentials
        # print(self.sequentials)
        # quit()
        self.sequential = self.make_sequential(units, dtype)

        # consider shared layers

    def make_sequential(self, units, dtype):
        dense = []
        for ii, n_hidden in enumerate(units):
            layer = Linear(
                n_hidden,
                w_init=self.w_init,
                b_init=self.b_init,
                use_ntk=self.use_ntk,
                dtype=dtype,
                name=f"dense_{ii}",
            )
            dense.append(layer)
            if ii < len(units) - 1:
                dense.append(swish)
        sequential = nn.Sequential(dense, name="readout")
        return sequential

    def __call__(self, x):
        # h = []
        
        # for seq in self.sequential:
        #     h.append(seq(x))
        h = self.sequential(x)

        if not self.is_feature_fn:
            h = jnp.reshape(h, (self.num_fidelities, self.n_shallow_ensemble))

        # h = jnp.concat(h, axis=-1) # which axis?
        # TODO should we move aggregation here?

        return h