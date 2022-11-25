from typing import Optional

import haiku as hk
import jax

from gmnn_jax.layers.activation import swish
from gmnn_jax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptor,
)
from gmnn_jax.layers.initializers import NTKBias, NTKWeights
from gmnn_jax.layers.scaling import PerElementScaleShift


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

        units = units + [1]
        self.dense = hk.nets.MLP(
            units,
            activation=swish,
            activate_final=False,
            w_init=NTKWeights(),
            b_init=NTKBias(),
            name="readout",
        )

        self.scale_shift = PerElementScaleShift(
            scale=2.0, shift=1.0, n_species=n_species, name="scale_shift"
        )

    def __call__(self, R, Z, neighbor):
        gm = self.descriptor(R, Z, neighbor)

        output = jax.vmap(self.dense)(gm)  # why is hk.vmap not required here?

        output = self.scale_shift(output, Z)

        return output
