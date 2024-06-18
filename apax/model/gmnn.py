import logging
from dataclasses import field
from typing import Any, Callable, Tuple, Union

import flax.linen as nn
import jax
import numpy as np
from jax import Array

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.distances import make_distance_fn
from apax.layers.empirical import EmpiricalEnergyTerm
from apax.layers.masking import mask_by_atom
from apax.layers.properties import stress_times_vol
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.layers.ntk_linear import NTKLinear
from apax.utils.jax_md_reduced import partition
from apax.utils.math import fp64_sum
from apax.layers.activation import swish

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


class AtomisticModel(nn.Module):
    """Most basic prediction model.
    Allesmbles descriptor, readout (NNs) and output scale-shifting.
    """

    descriptor: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    scale_shift: nn.Module = PerElementScaleShift()
    mask_atoms: bool = True

    def setup(self) -> None:
        self.lin_node_1 = NTKLinear(64, "lecun", "zeros", False)
        self.lin_message_1 = NTKLinear(64, "lecun", "zeros", False)


    def __call__(
        self,
        dr_vec: Array,
        Z: Array,
        idx: Array,
    ) -> Array:
        i, j = idx[0], idx[1]
        n_atoms = Z.shape[0]

        gm = self.descriptor(dr_vec, Z, idx)

        h = jax.vmap(self.lin_node_1)(gm)
        # linear
        h_i, h_j = h[i], h[j]
        h_j = swish(jax.vmap(self.lin_message_1)(h_j))

        u_i = jax.ops.segment_sum(h_j, j, n_atoms)
        h = h + u_i

        h = jax.vmap(self.readout)(h)
        output = self.scale_shift(h, Z)

        if self.mask_atoms:
            output = mask_by_atom(output, Z)
        return output



class FeatureModel(nn.Module):
    """Model wrapps some submodel (e.g. a descriptor) to supply distance computation."""

    feature_model: nn.Module = GaussianMomentDescriptor()
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    inference_disp_fn: Any = None

    def setup(self):
        self.compute_distances = make_distance_fn(self.init_box, self.inference_disp_fn)

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
        perturbation=None,
    ):
        dr_vec, idx = self.compute_distances(
            R,
            neighbor,
            box,
            offsets,
            perturbation,
        )

        features = self.feature_model(dr_vec, Z, idx)
        return features


class EnergyModel(nn.Module):
    """Model which post processes the output of an atomistic model and
    adds empirical energy terms.
    """

    atomistic_model: AtomisticModel = AtomisticModel()
    corrections: list[EmpiricalEnergyTerm] = field(default_factory=lambda: [])
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    inference_disp_fn: Any = None

    def setup(self):
        self.compute_distances = make_distance_fn(self.init_box, self.inference_disp_fn)

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
        perturbation=None,
    ):
        dr_vec, idx = self.compute_distances(
            R,
            neighbor,
            box,
            offsets,
            perturbation,
        )

        # Model Core
        atomic_energies = self.atomistic_model(dr_vec, Z, idx)
        total_energy = fp64_sum(atomic_energies)

        # Corrections
        for correction in self.corrections:
            energy_correction = correction(dr_vec, Z, idx)
            total_energy = total_energy + energy_correction

        # TODO think of nice abstraction for predicting additional properties
        return total_energy


class EnergyDerivativeModel(nn.Module):
    """Transforms an EnergyModel into one that also predicts derivatives the total energy.
    Can calculate forces and stress tensors.
    """

    # Alternatively, should this be a function transformation?
    energy_model: EnergyModel = EnergyModel()
    calc_stress: bool = False

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
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
