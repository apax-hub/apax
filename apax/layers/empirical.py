from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ase import data
from jax import vmap

from apax.layers.masking import mask_by_neighbor
from apax.utils.jax_md_reduced import space
from apax.utils.math import fp64_sum


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1.0)


class EmpiricalEnergyTerm(nn.Module):
    dtype: Any = jnp.float32


class ZBLRepulsion(EmpiricalEnergyTerm):
    r_max: float = 2.0
    apply_mask: bool = True

    def setup(self):
        self.distance = vmap(space.distance, 0, 0)

        coeffs = jnp.array([0.18175, 0.50986, 0.28022, 0.02817])[:, None]
        coeffs_isp = inverse_softplus(coeffs)
        rep_scale_isp = inverse_softplus(0.1)

        self.a_exp = 0.23
        self.a_num = 0.46850
        self.coefficients = self.param(
            "coefficients",
            nn.initializers.constant(coeffs_isp),
            (4, 1),
        )

        self.exponents = jnp.array([3.19980, 0.94229, 0.4029, 0.20162])[:, None]

        self.rep_scale = self.param(
            "rep_scale", nn.initializers.constant(rep_scale_isp), (1,)
        )

    def __call__(self, dr_vec, Z, idx):
        # Z shape n_atoms

        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(self.dtype)

        dr = jnp.clip(dr, a_min=0.02, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        a_exp = self.a_exp
        a_num = self.a_num
        coefficients = jax.nn.softplus(self.coefficients)
        exponents = self.exponents
        rep_scale = jax.nn.softplus(self.rep_scale)

        a_divisor = Z_i**a_exp + Z_j**a_exp
        dist = dr * a_divisor / a_num
        f = coefficients * jnp.exp(-exponents * dist)
        f = jnp.sum(f, axis=0)

        E_ij = Z_i * Z_j / dr * f * cos_cutoff
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx)
        E = 0.5 * rep_scale * fp64_sum(E_ij)
        return E


class ExponentialRepulsion(EmpiricalEnergyTerm):
    r_max: float = 2.0
    apply_mask: bool = True

    def setup(self):
        self.distance = vmap(space.distance, 0, 0)

        radii = data.covalent_radii * 0.8
        self.rscale = self.param("rep_scale", nn.initializers.constant(radii), (119,))


        self.prefactor = self.param(
            "rep_prefactor", nn.initializers.constant(10.0), (119,)
        )

    def __call__(self, dr_vec, Z, idx):
        # Z shape n_atoms

        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(self.dtype)

        dr = jnp.clip(dr, a_min=0.02, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        A_i, A_j = (
            0.1 * jax.numpy.abs(self.prefactor[Z_i]),
            0.1 * jax.numpy.abs(self.prefactor[Z_j]),
        )
        R_i, R_j = jax.numpy.abs(self.rscale[Z_i]), jax.numpy.abs(self.rscale[Z_j])

        f = A_i * A_j * jnp.exp(-dr * (R_i + R_j) / (R_i * R_j)) / dr**2

        E_ij = f * cos_cutoff
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx)
        E = fp64_sum(E_ij)
        return E


all_corrections = {
    "zbl" : ZBLRepulsion,
    "exponential" : ExponentialRepulsion,
}
