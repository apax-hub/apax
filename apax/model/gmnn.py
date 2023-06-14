import logging
from typing import Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
from jax_md import partition, quantity
from jax_md.util import Array

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.empirical import ReaxBonded, ZBLRepulsion
from apax.layers.masking import mask_by_atom
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.utils import NeighborSpoof
from apax.utils.math import fp64_sum

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


class AtomisticModel(nn.Module):
    descriptor: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    scale_shift: nn.Module = PerElementScaleShift()
    mask_atoms: bool = True

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, NeighborSpoof, Array],
        box,
        offsets,
        perturbation=None,
    ) -> Array:
        if type(neighbor) in [partition.NeighborList, NeighborSpoof]:
            idx = neighbor.idx
        else:
            idx = neighbor

        gm = self.descriptor(R, Z, idx, box, offsets, perturbation)
        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        if self.mask_atoms:
            output = mask_by_atom(output, Z)

        return output


class EnergyModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()
    repulsion: Optional[ZBLRepulsion] = None
    bonded: Optional[ReaxBonded] = None

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, NeighborSpoof, Array],
        box,
        offsets,
        perturbation=None,
    ):
        atomic_energies = self.atomistic_model(R, Z, neighbor, box, offsets, perturbation)
        total_energy = fp64_sum(atomic_energies)

        # Corrections
        if self.repulsion is not None:
            repulsion_energy = self.repulsion(R, Z, neighbor, box, perturbation)
            total_energy = total_energy + repulsion_energy

        if self.bonded is not None:
            bonded_energy = self.bonded(R, Z, neighbor, box, perturbation)
            total_energy = total_energy + bonded_energy
        return total_energy


import jax.numpy as jnp


def volume(dimension: int, box) -> float:
    if jnp.isscalar(box) or not box.ndim:
        return box**dimension
    elif box.ndim == 1:
        return jnp.prod(box)
    elif box.ndim == 2:
        return jnp.linalg.det(box)
    raise ValueError(f"Box must be either: a scalar, a vector, or a matrix. Found {box}.")


class EnergyDerivativeModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()
    repulsion: Optional[ZBLRepulsion] = None
    bonded: Optional[ReaxBonded] = None
    calc_stress: bool = False

    def setup(self):
        self.energy_fn = EnergyModel(
            atomistic_model=self.atomistic_model,
            repulsion=self.repulsion,
            bonded=self.bonded,
        )

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, NeighborSpoof, Array],
        box,
        offsets,
        perturbation=None,
    ):
        energy, neg_forces = jax.value_and_grad(self.energy_fn)(
            R, Z, neighbor, box, offsets, perturbation
        )
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}

        if self.calc_stress:
            stress = quantity.stress(
                lambda R, box, **kwargs: self.energy_fn(
                    R, Z, neighbor, box, offsets, **kwargs
                ),
                R,
                box,
            )
            dim = R.shape[1]
            vol_0 = volume(dim, box)
            prediction["stress"] = stress * vol_0

        return prediction
