from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

from gmnn_jax.layers.activation import swish
from gmnn_jax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptor,
)
from gmnn_jax.layers.scaling import PerElementScaleShift
from gmnn_jax.layers.compat_linear import CompatLinear


class GMNN(hk.Module):
    def __init__(
        self,
        units,
        displacement,
        n_basis=5,
        n_radial=4,
        n_species=10,
        n_atoms=3,
        r_min=0.5,
        r_max=6.0,
        use_all_features=True,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.descriptor = GaussianMomentDescriptor(
            displacement,
            n_basis,
            n_radial,
            n_species,
            n_atoms,
            r_min,
            r_max,
            use_all_features=use_all_features,
            name="descriptor",
        )

        # Hopefully we can soon go back to using a regular MLP
        # units = units + [1]
        # self.dense = hk.nets.MLP(
        #     units,
        #     activation=swish,
        #     activate_final=False,
        #     w_init=NTKWeights(),
        #     b_init=NTKBias(),
        #     name="readout",
        # )
        self.dense1 = CompatLinear(units[0],name="dense1")
        self.dense2 = CompatLinear(units[1],name="dense2")
        self.dense3 = CompatLinear(1,name="dense3")

        self.scale_shift = PerElementScaleShift(
            scale=2.0, shift=1.0, n_species=n_species, name="scale_shift"
        )

    def __call__(self, R, Z, neighbor):
        gm = self.descriptor(R, Z, neighbor)

        h = jax.vmap(self.dense1)(gm)  # why is hk.vmap not required here?
        h = swish(h)
        h = jax.vmap(self.dense2)(h)
        h = swish(h)
        h = jax.vmap(self.dense3)(h)

        output = self.scale_shift(h, Z)

        return output
