from dataclasses import field
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ase import data
from jax import vmap

from apax.layers.masking import mask_by_atom, mask_by_neighbor
from apax.utils.convert import str_to_dtype
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

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        dtype = str_to_dtype(self.dtype)
        # Z shape n_atoms

        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(dtype)

        dr = jnp.clip(dr, a_min=0.02, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        a_exp = self.a_exp
        a_num = self.a_num
        coefficients = jax.nn.softplus(self.coefficients)
        exponents = self.exponents
        rep_scale = jax.nn.softplus(self.rep_scale)[0]

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
            "rep_prefactor", nn.initializers.constant(100.0), (119,)
        )

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        dtype = str_to_dtype(self.dtype)

        # Z shape n_atoms
        idx_i, idx_j = idx[0], idx[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(dtype)

        dr = jnp.clip(dr, a_min=0.02, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr / self.r_max) + 1.0)

        # Ensure positive parameters
        A_i, A_j = (
            jax.numpy.abs(self.prefactor[Z_i]),
            jax.numpy.abs(self.prefactor[Z_j]),
        )
        R_i, R_j = jax.numpy.abs(self.rscale[Z_i]), jax.numpy.abs(self.rscale[Z_j])

        f = A_i * A_j * jnp.exp(-dr * (R_i + R_j) / (R_i * R_j)) / dr**2

        E_ij = f * cos_cutoff
        if self.apply_mask:
            E_ij = mask_by_neighbor(E_ij, idx)
        E = fp64_sum(E_ij)
        return E


class LatentEwald(EmpiricalEnergyTerm):
    """Latent Ewald summation by Cheng https://arxiv.org/abs/2408.15165
    Requires a property head which predicts 'charge' per atom.
    """

    kgrid: list[int] = field(default_factory=lambda: [2, 2, 2])
    sigma: float = 1.0
    apply_mask: bool = True

    def __call__(self, R, dr_vec, Z, idx, box, properties):
        # Z shape n_atoms
        if "charge" not in properties:
            raise KeyError(
                "property 'charge' not found. Make sure to predict it in the model section"
            )

        q = properties["charge"]

        V = jnp.linalg.det(box)
        Lx, Ly, Lz = jnp.linalg.norm(box, axis=1)

        k_range_x = 2 * np.pi * jnp.arange(1, self.kgrid[0]) / Lx
        k_range_y = 2 * np.pi * jnp.arange(1, self.kgrid[1]) / Ly
        k_range_z = 2 * np.pi * jnp.arange(1, self.kgrid[2]) / Lz

        kx, ky, kz = jnp.meshgrid(k_range_x, k_range_y, k_range_z)
        k = jnp.reshape(jnp.stack((kx, ky, kz), axis=-1), (-1, 3))

        k2 = jnp.sum(k**2, axis=-1)

        sf_k = q * jnp.exp(1j * jnp.einsum("id,jd->ij", R, k))
        if self.apply_mask:
            sf_k = mask_by_atom(sf_k, Z)
        sf = jnp.sum(sf_k, axis=0)
        S2 = jnp.abs(sf) ** 2

        E_lr = -fp64_sum(jnp.exp(-k2 * (self.sigma**2) / 2) / k2 * S2) / V

        return E_lr


all_corrections = {
    "zbl": ZBLRepulsion,
    "exponential": ExponentialRepulsion,
    "latent_ewald": LatentEwald,
}
