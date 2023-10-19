import logging
from dataclasses import field
from typing import Any, Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
from jax_md import partition, space
from jax_md.util import Array
import numpy as np
import jax.numpy as jnp
from jax import vmap

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.empirical import EmpiricalEnergyTerm
from apax.layers.masking import mask_by_atom
from apax.layers.properties import stress_times_vol
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.utils import NeighborSpoof
from apax.utils.math import fp64_sum

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)

def canonicalize_neighbors(neighbor):
    if type(neighbor) in [partition.NeighborList, NeighborSpoof]:
        idx = neighbor.idx
    else:
        idx = neighbor
    return idx


def disp_fn(ri, rj, perturbation, box):
    dR = space.pairwise_displacement(ri, rj)
    dR = space.transform(box, dR)

    if perturbation is not None:
        dR = dR + space.raw_transform(perturbation, dR)
        # https://github.com/mir-group/nequip/blob/c56f48fcc9b4018a84e1ed28f762fadd5bc763f1/nequip/nn/_grad_output.py#L267
        # https://github.com/sirmarcel/glp/blob/main/glp/calculators/utils.py
        # other codes do R = R + strain, not dR
        # can be implemented for efficiency
    return dR


def get_disp_fn(displacement):
    def disp_fn(ri, rj, perturbation, box):
        return displacement(ri, rj, perturbation, box=box)
    return disp_fn


class AtomisticModel(nn.Module):
    descriptor: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    scale_shift: nn.Module = PerElementScaleShift()
    mask_atoms: bool = True

    def __call__(
        self,
        dr_vec: Array,
        Z: Array,
        idx: Array,
        box: Array,
        offsets: Array,
        perturbation: Optional[Array]=None,
    ) -> Array:
        gm = self.descriptor(dr_vec, Z, idx, box, offsets, perturbation)
        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        if self.mask_atoms:
            output = mask_by_atom(output, Z)
        return output


class EnergyModel(nn.Module):
    atomistic_model: AtomisticModel = AtomisticModel()
    corrections: list[EmpiricalEnergyTerm] = field(default_factory=lambda: [])
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    inference_disp_fn: Any = None

    def setup(self):

        if np.all(self.init_box < 1e-6):
            # gas phase training and predicting
            displacement_fn = space.free()[0]
            self.displacement = space.map_bond(displacement_fn)
        elif self.inference_disp_fn is None:
            # for training on periodic systems
            self.displacement = vmap(disp_fn, (0, 0, None, None), 0)
        else:
            mappable_displacement_fn = get_disp_fn(self.inference_disp_fn)
            self.displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, NeighborSpoof, Array],
        box,
        offsets,
        perturbation=None,
    ):
        
        # Distances
        idx = canonicalize_neighbors(neighbor)
        idx_i, idx_j = idx[0], idx[1]

        # R shape n_atoms x 3
        R = R.astype(jnp.float64)
        Ri = R[idx_i]
        Rj = R[idx_j]

        # dr_vec shape: neighbors x 3
        if np.all(self.init_box < 1e-6):
            # reverse conventnion to match TF
            # distance vector for gas phase training and predicting
            dr_vec = self.displacement(Rj, Ri).astype(self.dtype)
        else:
            # distance vector for training on periodic systems
            dr_vec = self.displacement(Rj, Ri, perturbation, box).astype(self.dtype)
            dr_vec += offsets.astype(self.dtype)

        # Model Core
        atomic_energies = self.atomistic_model(dr_vec, Z, neighbor)
        total_energy = fp64_sum(atomic_energies)

        # Corrections
        for correction in self.corrections:
            energy_correction = correction(dr_vec, Z, neighbor)
            total_energy = total_energy + energy_correction

        # TODO think of nice abstraction for predicting additional properties
        return total_energy


class EnergyDerivativeModel(nn.Module):
    # Alternatively, should this be a function transformation?
    energy_model: EnergyModel = EnergyModel()
    corrections: list[EmpiricalEnergyTerm] = field(default_factory=lambda: [])
    calc_stress: bool = False

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, NeighborSpoof, Array],
        box,
        offsets,
    ):
        energy, neg_forces = jax.value_and_grad(self.energy_model)(
            R, Z, neighbor, box, offsets
        )
        forces = -neg_forces
        prediction = {"energy": energy, "forces": forces}

        if self.calc_stress:
            stress = stress_times_vol(
                self.energy_model, R, box, Z=Z, neighbor=neighbor, offsets=offsets
            )
            prediction["stress"] = stress

        return prediction
