from dataclasses import field
from typing import Any, Callable, List

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp
from jax.nn import swish

from apax.layers.descriptor.mace_blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
)
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
                dense.append(self.activation_fn)
        self.sequential = nn.Sequential(dense, name="readout")

    def __call__(self, x):
        h = self.sequential(x)
        # TODO should we move aggregation here?
        return h


class MaceReadout(nn.Module):
    """Per-layer readout sum matching the foundation MACE forward pass.

    Designed for the readout slot in :class:`apax.nn.models.EnergyModel`.
    ``EnergyModel`` vmaps the readout over atoms, so each invocation sees a
    single atom's flat per-layer-concatenated feature vector. The readout
    splits the flat vector back into per-layer scalar chunks, applies a
    :class:`~apax.layers.descriptor.mace_blocks.LinearReadoutBlock` to layers
    ``0..num_interactions-2`` and a
    :class:`~apax.layers.descriptor.mace_blocks.NonLinearReadoutBlock` to the
    last layer, and returns the sum.

    Parameters
    ----------
    num_interactions : int
        Number of interaction layers in the MACE backbone (and the number of
        per-layer scalar chunks expected in the input).
    hidden_dim : int
        Per-layer scalar channel count; equals the ``0e`` dimension of
        ``MaceRepresentation.hidden_irreps``.
    MLP_irreps : str, default = "16x0e"
        Hidden irreps of the last-layer non-linear MLP. Must be scalar-only.
    n_shallow_ensemble : int, default = 0
        When > 0, each block's final projection emits ``n_shallow_ensemble``
        scalars. Downstream :class:`~apax.nn.models.EnergyModel` auto-detects
        the ensemble case from ``E_i.shape[1] > 1``.
    dtype : Any, default = jnp.float32
        Floating-point dtype for internal computations.
    """

    num_interactions: int
    hidden_dim: int
    MLP_irreps: str = "16x0e"
    n_shallow_ensemble: int = 0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """Return a per-atom energy summed across MACE layers.

        Parameters
        ----------
        x : jnp.ndarray, shape ``(num_interactions * hidden_dim,)``
            Flat per-atom feature vector after vmap.

        Returns
        -------
        jnp.ndarray, shape ``(1,)`` or ``(n_shallow_ensemble,)``
            Per-atom scalar (or ensemble of scalars).
        """
        dtype = str_to_dtype(self.dtype)
        x = x.astype(dtype)
        layers = x.reshape(self.num_interactions, self.hidden_dim)
        n_out = self.n_shallow_ensemble if self.n_shallow_ensemble > 0 else 1

        E = jnp.zeros((n_out,), dtype=dtype)
        for k in range(self.num_interactions):
            feat = e3nn.IrrepsArray(f"{self.hidden_dim}x0e", layers[k])
            if k < self.num_interactions - 1:
                contrib = LinearReadoutBlock(n_out=n_out, name=f"readout_{k}")(feat)
            else:
                contrib = NonLinearReadoutBlock(
                    MLP_irreps=self.MLP_irreps,
                    n_out=n_out,
                    name=f"readout_{k}",
                )(feat)
            self.sow("debug", f"readouts[{k}]", contrib)
            E = E + contrib
        return E
