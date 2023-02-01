import dataclasses
import logging
from typing import Callable, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition
from jax_md.util import Array, high_precision_sum

from gmnn_jax.layers.activation import swish
from gmnn_jax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptor,
)
from gmnn_jax.layers.ntk_linear import NTKLinear
from gmnn_jax.layers.scaling import PerElementScaleShift
from gmnn_jax.layers.masking import mask_by_atom

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


class GMNN(hk.Module):
    def __init__(
        self,
        units: List[int],
        displacement: DisplacementFn,
        n_atoms: int,
        n_basis: int = 7,
        n_radial: int = 5,
        n_species: int = 119,
        r_min: float = 0.5,
        r_max: float = 6.0,
        b_init: str = "normal",
        elemental_energies_mean: Optional[Array] = None,
        elemental_energies_std: Optional[Array] = None,
        apply_mask: bool = True,
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
            apply_mask,
            name="descriptor",
        )

        units = units + [1]
        dense = []
        for ii, n_hidden in enumerate(units):
            dense.append(NTKLinear(n_hidden, b_init=b_init, name=f"dense_{ii}"))
            if ii < len(units) - 1:
                dense.append(swish)
        self.readout = hk.Sequential(dense, name="readout")

        self.scale_shift = PerElementScaleShift(
            scale=elemental_energies_std,
            shift=elemental_energies_mean,
            n_species=n_species,
            name="scale_shift",
        )

        self.apply_mask = apply_mask

    def __call__(self, R: Array, Z: Array, neighbor: partition.NeighborList) -> Array:
        gm = self.descriptor(R, Z, neighbor)
        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        if self.apply_mask:
            output = mask_by_atom(output, Z)

        return output


def get_md_model(
    atomic_numbers: Array,
    displacement: DisplacementFn,
    nn: List[int] = [512, 512],
    box_size: float = 100.0,
    r_max: float = 6.0,
    n_basis: int = 7,
    n_radial: int = 5,
    dr_threshold: float = 0.5,
    nl_format: partition.NeighborListFormat = partition.Sparse,
    **neighbor_kwargs,
) -> MDModel:
    neighbor_fn = partition.neighbor_list(
        displacement,
        box_size,
        r_max,
        dr_threshold,
        fractional_coordinates=False,
        format=nl_format,
        **neighbor_kwargs,
    )

    n_atoms = atomic_numbers.shape[0]
    Z = jnp.asarray(atomic_numbers)
    # casting ot python int prevents n_species from becoming a tracer,
    # which causes issues in the NVT `apply_fn`
    n_species = int(np.max(Z) + 1)

    @hk.without_apply_rng
    @hk.transform
    def model(R, neighbor):
        gmnn = GMNN(
            nn,
            displacement,
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_radial=n_radial,
            n_species=n_species,
            r_max=r_max,
        )
        out = gmnn(R, Z, neighbor)
        return high_precision_sum(out)

    return neighbor_fn, model


@dataclasses.dataclass
class NeighborSpoof:
    idx: jnp.array


def get_training_model(
    n_atoms: int,
    n_species: int,
    displacement_fn: DisplacementFn,
    nn: List[int],
    n_basis: int = 7,
    n_radial: int = 5,
    r_min: float = 0.5,
    r_max: float = 6.0,
    b_init: str = "normal",
    elemental_energies_mean: Optional[Array] = None,
    elemental_energies_std: Optional[Array] = None,
) -> Tuple[Callable, Callable]:
    log.info("Bulding Model")

    @hk.without_apply_rng
    @hk.transform
    def model(R, Z, idx):
        gmnn = GMNN(
            nn,
            displacement_fn,
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_radial=n_radial,
            n_species=n_species,
            r_min=r_min,
            r_max=r_max,
            b_init=b_init,
            elemental_energies_mean=elemental_energies_mean,
            elemental_energies_std=elemental_energies_std,
        )
        neighbor = NeighborSpoof(idx)

        def energy_fn(R, Z, neighbor):
            out = gmnn(R, Z, neighbor)
            # mask = partition.neighbor_list_mask(neighbor)
            # out = out * mask
            energy = high_precision_sum(out)
            return energy

        energy, neg_forces = jax.value_and_grad(energy_fn)(R, Z, neighbor)
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}
        return prediction

    return model
