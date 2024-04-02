from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from apax.layers.masking import mask_by_neighbor
from apax.utils.jax_md_reduced import space
from apax.utils.math import fp64_sum


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1.0)


class EmpiricalEnergyTerm(nn.Module):
    dtype: Any = jnp.float32


class ZBLRepulsion(EmpiricalEnergyTerm):
    r_max: float = 6.0
    apply_mask: bool = True

    def setup(self):
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
