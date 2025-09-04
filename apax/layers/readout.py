from dataclasses import field
from typing import Any, Callable, List

import flax.linen as nn
import jax.numpy as jnp

from apax.layers.activation import swish
from apax.layers.ntk_linear import NTKLinear
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
            layer = NTKLinear(
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


class GatingNetwork(nn.Module):
    units: List[int] = field(default_factory=lambda: [32, 32])
    num_experts: int = 4
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = True
    n_shallow_ensemble: int = 0
    is_feature_fn: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        units = list(self.units)
        readout_unit = [self.num_experts]
        units += readout_unit

        dtype = str_to_dtype(self.dtype)

        dense = []
        for ii, n_hidden in enumerate(units):
            layer = NTKLinear(
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
        self.sequential = nn.Sequential(dense, name="gating")

    def __call__(self, x):
        logits = self.sequential(x)
        return nn.softmax(logits)


class MoEReadout(nn.Module):
    units: List[int] = field(default_factory=lambda: [32, 32])
    gating_units: List[int] = field(default_factory=lambda: [32, 32])
    num_experts: int = 4
    activation_fn: Callable = swish
    w_init: str = "normal"
    b_init: str = "zeros"
    use_ntk: bool = True
    n_shallow_ensemble: int = 0
    is_feature_fn: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        dtype = str_to_dtype(self.dtype)
        gating_fn = GatingNetwork(units=self.gating_units, num_experts=self.num_experts, dtype=dtype)
        experts = [
            AtomisticReadout(
                units=self.units,
                activation_fn=self.activation_fn,
                w_init=self.w_init,
                b_init=self.b_init,
                use_ntk=self.use_ntk,
                n_shallow_ensemble=self.n_shallow_ensemble,
                dtype=dtype
            ) for _ in range(self.num_experts)
         ]
        gate_values = gating_fn(x)
        expert_outputs = jnp.stack([expert(x) for expert in experts], axis=0)[:,0]
        output = jnp.sum(gate_values * expert_outputs, axis=-1, keepdims=True)
        # TODO check for compatibility with shallow ensemble
        return output