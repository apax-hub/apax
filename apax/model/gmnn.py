import logging
from typing import Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
from jax_md import partition
from jax_md.util import Array

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.empirical import ReaxBonded, ZBLRepulsion
from apax.layers.masking import mask_by_atom
from apax.layers.properties import stress_times_vol
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
            repulsion_energy = self.repulsion(R, Z, neighbor, box, offsets, perturbation)
            total_energy = total_energy + repulsion_energy

        if self.bonded is not None:
            bonded_energy = self.bonded(R, Z, neighbor, box, offsets, perturbation)
            total_energy = total_energy + bonded_energy
        return total_energy


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
    ):
        energy, neg_forces = jax.value_and_grad(self.energy_fn)(
            R, Z, neighbor, box, offsets
        )
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}

        if self.calc_stress:
            stress = stress_times_vol(
                self.energy_fn, R, box, Z=Z, neighbor=neighbor, offsets=offsets
            )
            prediction["stress"] = stress

        return prediction
