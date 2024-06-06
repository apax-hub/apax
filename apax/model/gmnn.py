import logging
from dataclasses import field
from typing import Any, Callable, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, vmap

from apax.layers.descriptor.gaussian_moment_descriptor import GaussianMomentDescriptor
from apax.layers.empirical import EmpiricalEnergyTerm
from apax.layers.masking import get_neighbor_mask, get_node_mask, mask_by_atom
from apax.layers.properties import stress_times_vol
from apax.layers.readout import AtomisticReadout
from apax.layers.scaling import PerElementScaleShift
from apax.model.myrto.so3krates.so3krates import So3krates
from apax.utils.jax_md_reduced import partition, space
from apax.utils.math import fp64_sum

DisplacementFn = Callable[[Array, Array], Array]
MDModel = Tuple[partition.NeighborFn, Callable, Callable]

log = logging.getLogger(__name__)


def canonicalize_neighbors(neighbor):
    return neighbor.idx if isinstance(neighbor, partition.NeighborList) else neighbor


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
    

class AtomicSo3krates(nn.Module):
    cutoff: float = 5.0
    num_layers: int = 1
    max_degree: int = 3
    num_features: int = 128
    num_radial_features: int = 32
    num_heads: int = 4
    use_layer_norm_1: bool = False
    use_layer_norm_2: bool = False
    use_layer_norm_final: bool = False
    activation: str = "silu"
    cutoff_fn: str = "cosine_cutoff"
    transform_input_features: bool = False
    scale_shift: nn.Module = PerElementScaleShift()

    @nn.compact
    def __call__(self,
        dr_vec: Array,
        Z: Array,
        idx: Array,
        ):

        model = So3krates(
            cutoff=self.cutoff,
            num_layers=self.num_layers,
            max_degree=self.max_degree,
            num_features=self.num_features,
            num_radial_features=self.num_radial_features,
            num_heads=self.num_heads,
            use_layer_norm_1=self.use_layer_norm_1,
            use_layer_norm_2=self.use_layer_norm_2,
            use_layer_norm_final=self.use_layer_norm_final,
            activation=self.activation,
            cutoff_fn=self.cutoff_fn,
            transform_input_features=self.transform_input_features,
        )

        R_ij = dr_vec
        i,j = idx[0], idx[1]
        Z_i = Z

        pair_mask = get_neighbor_mask(idx)
        node_mask = get_node_mask(Z)

        h = model(
            R_ij,
            i,
            j,
            Z_i,
            pair_mask,
            node_mask,
        )
        output = self.scale_shift(h, Z)
        return output


class EnergyModel(nn.Module):
    """Model which post processes the output of an atomistic model and
    adds empirical energy terms.
    """

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
        neighbor: Union[partition.NeighborList, Array],
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
            dr_vec = self.displacement(Rj, Ri)
        else:
            # distance vector for training on periodic systems
            dr_vec = self.displacement(Rj, Ri, perturbation, box)
            dr_vec += offsets

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
