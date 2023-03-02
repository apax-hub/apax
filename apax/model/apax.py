import dataclasses
import logging
from typing import Callable, List, Optional, Tuple

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition
from jax_md.util import Array, high_precision_sum

from apax.layers.activation import swish
from apax.layers.descriptor.gaussian_moment_descriptor import (
    GaussianMomentDescriptor,
    GaussianMomentDescriptorFlax,
)
from apax.layers.masking import mask_by_atom
from apax.layers.ntk_linear import NTKLinear
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift, PerElementScaleShiftFlax
from apax.utils.math import fp64_sum

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


class APAX(hk.Module):
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
        init_box: Optional[np.array] = np.array([0.0, 0.0, 0.0]),
        descriptor_dtype=jnp.float32,
        readout_dtype=jnp.float32,
        scale_shift_dtype=jnp.float32,
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
            init_box=init_box,
            dtype=descriptor_dtype,
            name="descriptor",
        )
        units = units + [1]
        dense = []
        for ii, n_hidden in enumerate(units):
            dense.append(
                NTKLinear(
                    n_hidden, b_init=b_init, dtype=readout_dtype, name=f"dense_{ii}"
                )
            )
            if ii < len(units) - 1:
                dense.append(swish)
        self.readout = hk.Sequential(dense, name="readout")

        self.scale_shift = PerElementScaleShift(
            scale=elemental_energies_std,
            shift=elemental_energies_mean,
            n_species=n_species,
            dtype=scale_shift_dtype,
            name="scale_shift",
        )

        self.scale_shift_dtype = scale_shift_dtype

        self.apply_mask = apply_mask

    def __call__(
        self, R: Array, Z: Array, neighbor: partition.NeighborList, box
    ) -> Array:
        gm = self.descriptor(R, Z, neighbor, box)

        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        assert output.dtype == self.scale_shift_dtype
        if self.apply_mask:
            output = mask_by_atom(output, Z)

        return output


class AtomisticModel(nn.Module):
    descriptor: nn.Module = GaussianMomentDescriptorFlax()
    readout: nn.Module = AtomisticReadout()
    scale_shift: nn.Module = PerElementScaleShiftFlax()
    mask_atoms: bool = True

    def __call__(
        self, R: Array, Z: Array, neighbor: partition.NeighborList, box
    ) -> Array:
        gm = self.descriptor(R, Z, neighbor.idx, box)
        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        if self.mask_atoms:
            output = mask_by_atom(output, Z)

        return output


class EnergyModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()

    def __call__(self, R: Array, Z: Array, neighbor: partition.NeighborList, box):
        atomic_energies = self.atomistic_model(R, Z, neighbor, box=box)
        total_energy = fp64_sum(atomic_energies)
        return total_energy


class EnergyForceModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()

    def __call__(self, R: Array, Z: Array, idx: Array, box):
        neighbor = NeighborSpoof(idx)

        def energy_fn(R, Z, neighbor, box):
            atomic_energies = self.atomistic_model(R, Z, neighbor, box=box)
            total_energy = fp64_sum(atomic_energies)
            return total_energy

        energy, neg_forces = jax.value_and_grad(energy_fn)(R, Z, neighbor, box)
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}
        return prediction


def get_md_model(
    atomic_numbers: Array,
    displacement: DisplacementFn,
    nn: List[int] = [512, 512],
    box: Optional[np.array] = np.array([0.0, 0.0, 0.0]),
    r_max: float = 6.0,
    n_basis: int = 7,
    n_radial: int = 5,
    dr_threshold: float = 0.5,
    nl_format: partition.NeighborListFormat = partition.Sparse,
    descriptor_dtype=jnp.float32,
    readout_dtype=jnp.float32,
    scale_shift_dtype=jnp.float32,
    **neighbor_kwargs,
) -> MDModel:
    default_box = 100
    if np.all(box < 1e-6):
        box = default_box

    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
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
        apax = APAX(
            nn,
            displacement,
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_radial=n_radial,
            n_species=n_species,
            r_max=r_max,
            descriptor_dtype=descriptor_dtype,
            readout_dtype=readout_dtype,
            scale_shift_dtype=scale_shift_dtype,
        )
        out = apax(R, Z, neighbor, jnp.array([0.0, 0.0, 0.0]))
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
    init_box: Optional[np.array] = np.array([0.0, 0.0, 0.0]),
    descriptor_dtype=jnp.float32,
    readout_dtype=jnp.float32,
    scale_shift_dtype=jnp.float32,
) -> Tuple[Callable, Callable]:
    log.info("Bulding Model")

    @hk.without_apply_rng
    @hk.transform
    def model(R, Z, idx, box):
        apax = APAX(
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
            init_box=init_box,
            descriptor_dtype=descriptor_dtype,
            readout_dtype=readout_dtype,
            scale_shift_dtype=scale_shift_dtype,
        )
        neighbor = NeighborSpoof(idx)

        def energy_fn(R, Z, neighbor, box):
            out = apax(R, Z, neighbor, box)
            # mask = partition.neighbor_list_mask(neighbor)
            # out = out * mask
            energy = high_precision_sum(out)
            return energy

        energy, neg_forces = jax.value_and_grad(energy_fn)(R, Z, neighbor, box)
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}
        return prediction

    return model
