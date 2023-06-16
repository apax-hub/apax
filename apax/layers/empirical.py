from typing import Any

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_md import partition, space

from apax.layers.descriptor.gaussian_moment_descriptor import disp_fn, get_disp_fn
from apax.layers.initializers import uniform_range
from apax.layers.masking import mask_by_neighbor
from apax.model.utils import NeighborSpoof
from apax.utils.math import fp64_sum


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1.0)


class ZBLRepulsion(nn.Module):
    dtype: Any = jnp.float32
    init_box: np.array = np.array([0.0, 0.0, 0.0])
    r_max: float = 6.0
    apply_mask: bool = True
    inference_disp_fn = None

    def setup(self):
        if np.all(self.init_box < 1e-6):
            # displacement function for gas phase training and predicting
            displacement_fn = space.free()[0]
            self.displacement = space.map_bond(displacement_fn)
        elif self.inference_disp_fn is None:
            # displacement function used for training on periodic systems
            self.displacement = vmap(disp_fn, (0, 0, None, None), 0)
        else:
            mappable_displacement_fn = get_disp_fn(self.inference_disp_fn)
            self.displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

        self.distance = vmap(space.distance, 0, 0)

        self.ke = 14.3996

        a_exp = 0.23
        a_num = 0.46850
        coeffs = jnp.array([0.18175, 0.50986, 0.28022, 0.02817])[:, None]
        exps = jnp.array([3.19980, 0.94229, 0.4029, 0.20162])[:, None]

        a_exp_isp = inverse_softplus(a_exp)
        a_num_isp = inverse_softplus(a_num)
        coeffs_isp = inverse_softplus(coeffs)
        exps_isp = inverse_softplus(exps)
        rep_scale_isp = inverse_softplus(1.0 / self.ke)

        self.a_exp = self.param("a_exp", nn.initializers.constant(a_exp_isp), (1,))
        self.a_num = self.param("a_num", nn.initializers.constant(a_num_isp), (1,))
        self.coefficients = self.param(
            "coefficients",
            nn.initializers.constant(coeffs_isp),
            (4, 1),
        )

        self.exponents = self.param(
            "exponents",
            nn.initializers.constant(exps_isp),
            (4, 1),
        )

        self.rep_scale = self.param(
            "rep_scale", nn.initializers.constant(rep_scale_isp), (1,)
        )

    def __call__(
        self, R, Z, neighbor, box, perturbation=None
    ):  # TODO how can you calc distancies without offset?
        R = R.astype(jnp.float64)
        # R shape n_atoms x 3
        # Z shape n_atoms
        # n_atoms = R.shape[0]
        if type(neighbor) in [partition.NeighborList, NeighborSpoof]:
            idx = neighbor.idx
        else:
            idx = neighbor

        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr_vec shape: neighbors x 3
        if np.all(self.init_box < 1e-6):
            # reverse conventnion to match TF
            # distance vector for gas phase training and predicting
            dr_vec = self.displacement(R[idx_j], R[idx_i]).astype(self.dtype)
        else:
            # distance vector for training on periodic systems
            # reverse conventnion to match TF
            Ri = R[idx_i]
            Rj = R[idx_j]

            dr_vec = self.displacement(Rj, Ri, perturbation, box).astype(self.dtype)
            # dr_vec -= offsets

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(self.dtype)

        dr = jnp.clip(dr, a_min=0.02, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        a_exp = jax.nn.softplus(self.a_exp)
        a_num = jax.nn.softplus(self.a_num)
        coefficients = jax.nn.softplus(self.coefficients)
        exponents = jax.nn.softplus(self.exponents)
        rep_scale = jax.nn.softplus(self.rep_scale)

        a_divisor = Z_i**a_exp + Z_j**a_exp
        dist = dr * a_divisor / a_num
        f = coefficients * jnp.exp(-exponents * dist)
        f = jnp.sum(f, axis=0)

        E_ij = Z_i * Z_j / dr * f * cos_cutoff
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx)
        E = 0.5 * rep_scale * self.ke * fp64_sum(E_ij)
        return fp64_sum(E)


class ReaxBonded(nn.Module):
    dtype: Any = jnp.float64
    n_species: int = 119
    init_box: np.array = np.array([0.0, 0.0, 0.0])
    r_max: float = 6.0
    apply_mask: bool = True
    inference_disp_fn = None

    def setup(self):
        if np.all(self.init_box < 1e-6):
            # displacement function for gas phase training and predicting
            displacement_fn = space.free()[0]
            self.displacement = space.map_bond(displacement_fn)
        elif self.inference_disp_fn is None:
            # displacement function used for training on periodic systems
            self.displacement = vmap(disp_fn, (0, 0, None, None), 0)
        else:
            mappable_displacement_fn = get_disp_fn(self.inference_disp_fn)
            self.displacement = vmap(mappable_displacement_fn, (0, 0, None, None), 0)

        self.distance = vmap(space.distance, 0, 0)

        self.r0 = self.param("r0", nn.initializers.constant(1.7), (self.n_species))
        self.po_coeff = self.param(
            "po_coeff", nn.initializers.uniform(0.9), (self.n_species, self.n_species, 3)
        )
        self.po_exp = self.param(
            "po_exp", uniform_range(1, 2), (self.n_species, self.n_species, 3)
        )
        self.De = self.param(
            "De", uniform_range(0.01, 0.05), (self.n_species, self.n_species)
        )
        self.pbe1 = self.param(
            "pbe1", uniform_range(-0.10, 0.10), (self.n_species, self.n_species)
        )
        self.pbe2 = self.param(
            "pbe2", uniform_range(0.0, 1.0), (self.n_species, self.n_species)
        )

    def __call__(
        self, R, Z, neighbor, box, perturbation=None
    ):  # TODO how can you calc distancies without offset?
        R = R.astype(jnp.float64)
        # R shape n_atoms x 3
        # Z shape n_atoms
        if type(neighbor) in [partition.NeighborList, NeighborSpoof]:
            idx = neighbor.idx
        else:
            idx = neighbor

        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr_vec shape: neighbors x 3
        if np.all(self.init_box < 1e-6):
            # reverse conventnion to match TF
            # distance vector for gas phase training and predicting
            dr_vec = self.displacement(R[idx_j], R[idx_i]).astype(self.dtype)
        else:
            # distance vector for training on periodic systems
            # reverse conventnion to match TF
            Ri = R[idx_i]
            Rj = R[idx_j]

            dr_vec = self.displacement(Rj, Ri, perturbation, box).astype(self.dtype)
            # dr_vec -= offsets

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(self.dtype)

        dr_clipped = jnp.clip(dr, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr_clipped / self.r_max) + 1.0)
        cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")

        r0_i, r0_j = jax.nn.softplus(self.r0[Z_i]), jax.nn.softplus(self.r0[Z_j])
        po_coeff = jax.nn.softplus(self.po_coeff[Z_i, Z_j])
        po_exp = jax.nn.softplus(self.po_exp[Z_i, Z_j])
        De = jax.nn.softplus(self.De[Z_i, Z_j])
        pbe1 = self.pbe1[Z_i, Z_j]
        pbe2 = jax.nn.softplus(self.pbe2[Z_i, Z_j] + 1.0)

        r0_ij = (r0_i + r0_j) / 2

        bo_terms = jnp.exp(-po_coeff * ((dr / r0_ij)[:, None] ** po_exp)) * cutoff
        bo_ij = jnp.sum(bo_terms, axis=1)

        E_ij = -De * bo_ij * jnp.exp(pbe1 * (1 - (bo_ij**pbe2)))

        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx)
        E = fp64_sum(E_ij)

        return E
