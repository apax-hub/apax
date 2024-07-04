import logging
from dataclasses import field
from typing import Any, Callable, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.distances import make_distance_fn
from apax.layers.empirical import EmpiricalEnergyTerm
from apax.layers.masking import mask_by_atom
from apax.layers.properties import stress_times_vol
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.utils.jax_md_reduced import partition
from apax.utils.math import fp64_sum

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

    def __call__(
        self,
        dr_vec: Array,
        Z: Array,
        idx: Array,
    ) -> Array:
        gm = self.descriptor(dr_vec, Z, idx)
        h = jax.vmap(self.readout)(gm)
        output = self.scale_shift(h, Z)

        if self.mask_atoms:
            output = mask_by_atom(output, Z)
        return output


class FeatureModel(nn.Module):
    """Model wrapps some submodel (e.g. a descriptor) to supply distance computation."""

    # atomistic_model: nn.Module = GaussianMomentDescriptor()
    descriptor: nn.Module = GaussianMomentDescriptor()
    readout: nn.Module = AtomisticReadout()
    should_average: bool = False
    init_box: np.array = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    inference_disp_fn: Any = None
    mask_atoms: bool = True

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

        # features = self.atomistic_model(dr_vec, Z, idx)
        gm = self.descriptor(dr_vec, Z, idx)
        features = jax.vmap(self.readout)(gm)

        if self.mask_atoms:
            features = mask_by_atom(features, Z)
        if self.should_average:
            features = jnp.mean(features, axis=0)
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
        # shape Natoms
        # shape shallow ens: Natoms x Nensemble
        atomic_energies = self.atomistic_model(dr_vec, Z, idx)

        # check for shallow ensemble
        is_shallow_ensemble = atomic_energies.shape[1] > 1
        if is_shallow_ensemble:
            total_energies_ensemble = fp64_sum(atomic_energies, axis=0)
            # shape Nensemble
            result = total_energies_ensemble
        else:
            # shape ()
            result = fp64_sum(atomic_energies)

        # Corrections
        for correction in self.corrections:
            energy_correction = correction(dr_vec, Z, idx)
            result = result + energy_correction

        # TODO think of nice abstraction for predicting additional properties
        return result


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


def make_mean_energy_fn(energy_fn):
    def mean_energy_fn(
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
        perturbation=None,
    ):
        e_ens = energy_fn(R, Z, neighbor, box, offsets, perturbation)
        E_mean = jnp.mean(e_ens)
        return E_mean

    return mean_energy_fn


class ShallowEnsembleModel(nn.Module):
    """Transforms an EnergyModel into one that also predicts derivatives the total energy.
    Can calculate forces and stress tensors.
    """

    energy_model: EnergyModel = EnergyModel()
    calc_stress: bool = False
    force_variance: bool = True

    def __call__(
        self,
        R: Array,
        Z: Array,
        neighbor: Union[partition.NeighborList, Array],
        box,
        offsets,
    ):
        energy_ens = self.energy_model(R, Z, neighbor, box, offsets)
        mean_energy_fn = make_mean_energy_fn(self.energy_model)

        n_ens = energy_ens.shape[0]
        divisor = 1 / (n_ens - 1)
        energy_mean = jnp.mean(energy_ens)
        energy_variance = divisor * fp64_sum((energy_ens - energy_mean) ** 2)

        prediction = {
            "energy": energy_mean,
            "energy_ensemble": energy_ens,
            "energy_uncertainty": jnp.sqrt(energy_variance),
        }

        if self.force_variance:
            forces_ens = -jax.jacrev(self.energy_model)(R, Z, neighbor, box, offsets)
            forces_mean = jnp.mean(forces_ens, axis=0)
            forces_variance = divisor * fp64_sum((forces_ens - forces_mean) ** 2, axis=0)

            prediction["forces"] = forces_mean
            prediction["forces_uncertainty"] = jnp.sqrt(forces_variance)
            prediction["forces_ensemble"] = forces_ens
        else:
            forces_mean = -jax.grad(mean_energy_fn)(R, Z, neighbor, box, offsets)
            prediction["forces"] = forces_mean

        if self.calc_stress:
            stress = stress_times_vol(
                mean_energy_fn, R, box, Z=Z, neighbor=neighbor, offsets=offsets
            )
            prediction["stress"] = stress

        return prediction
