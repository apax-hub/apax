from typing import Callable, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from jax_md import partition
from jax_md.util import Array, high_precision_sum

from gmnn_jax.layers.activation import swish
from gmnn_jax.layers.compat_linear import CompatLinear
from gmnn_jax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptor,
)
from gmnn_jax.layers.scaling import PerElementScaleShift

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]


class GMNN(hk.Module):
    def __init__(
        self,
        units: List[int],
        displacement: DisplacementFn,
        n_basis: int = 5,
        n_radial: int = 4,
        n_species: int = 10,
        n_atoms: int = 3,
        r_min: float = 0.5,
        r_max: float = 6.0,
        use_all_features: bool = True,
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
        self.dense1 = CompatLinear(units[0], name="dense1")
        self.dense2 = CompatLinear(units[1], name="dense2")
        self.dense3 = CompatLinear(1, name="dense3")

        self.scale_shift = PerElementScaleShift(
            scale=2.0, shift=1.0, n_species=n_species, name="scale_shift"
        )

    def __call__(self, R: Array, Z: Array, neighbor: partition.NeighborList) -> Array:
        gm = self.descriptor(R, Z, neighbor)

        h = jax.vmap(self.dense1)(gm)  # why is hk.vmap not required here?
        h = swish(h)
        h = jax.vmap(self.dense2)(h)
        h = swish(h)
        h = jax.vmap(self.dense3)(h)

        output = self.scale_shift(h, Z)

        return output


def get_gmnn_model(
    atomic_numbers: Array,
    units: List[int],
    displacement: DisplacementFn,
    box_size: float = 10.0,
    cutoff_distance: float = 6.0,
    n_basis: int = 7,
    n_radial: int = 5,
    dr_threshold: float = 0.5,
    nl_format: partition.NeighborListFormat = partition.Sparse,
    **neighbor_kwargs
) -> MDModel:
    neighbor_fn = partition.neighbor_list(
        displacement,
        box_size,
        cutoff_distance,
        dr_threshold,
        fractional_coordinates=False,
        format=nl_format,
        **neighbor_kwargs
    )

    n_atoms = atomic_numbers.shape[0]
    Z = jnp.asarray(atomic_numbers)
    n_species = jnp.max(Z)

    @hk.without_apply_rng
    @hk.transform
    def model(R, neighbor):
        gmnn = GMNN(
            units,
            displacement,
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_radial=n_radial,
            n_species=n_species,
        )
        out = gmnn(R, Z, neighbor)
        # mask = partition.neighbor_list_mask(neighbor)
        # out = out * mask
        return high_precision_sum(out)  # jnp.sum(out)

    return neighbor_fn, model.init, model.apply
