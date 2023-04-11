import dataclasses
import logging
from typing import Callable, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_md import partition
from jax_md.util import Array

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.masking import mask_by_atom
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.utils.math import fp64_sum

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


@dataclasses.dataclass
class NeighborSpoof:
    idx: jnp.array


class AtomisticModel(nn.Module):
    descriptor: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    scale_shift: nn.Module = PerElementScaleShift()
    mask_atoms: bool = True

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
    ) -> Array:
        if type(neighbor) in [partition.NeighborList, NeighborSpoof]:
            idx = neighbor.idx
        else:
            idx = neighbor

        gm = self.descriptor(R, Z, idx, box, offsets)
        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        if self.mask_atoms:
            output = mask_by_atom(output, Z)

        return output


class EnergyModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()

    def __call__(
        self, R: Array, Z: Array, neighbor: partition.NeighborList, box, offsets
    ):
        atomic_energies = self.atomistic_model(R, Z, neighbor, box=box, offsets=offsets)
        total_energy = fp64_sum(atomic_energies)
        return total_energy


class EnergyForceModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()

    def __call__(self, R: Array, Z: Array, idx: Array, box, offsets):
        neighbor = NeighborSpoof(idx)

        def energy_fn(R, Z, neighbor, box, offsets):
            atomic_energies = self.atomistic_model(
                R, Z, neighbor, box=box, offsets=offsets
            )
            total_energy = fp64_sum(atomic_energies)
            return total_energy

        energy, neg_forces = jax.value_and_grad(energy_fn)(R, Z, neighbor, box, offsets)
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}
        return prediction
