from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from apax.layers.initializers import uniform_range
from apax import ops


def gaussian_basis_impl(dr, shifts, betta, rad_norm):
    dr = einops.repeat(dr, "neighbors -> neighbors 1")
    # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
    distances = shifts - dr

    # shape: neighbors x n_basis
    basis = ops.exp(-betta * (distances**2))
    basis = rad_norm * basis
    return basis


def cosine_cutoff(dr, r_max):
    # shape: neighbors
    dr_clipped = ops.clip(dr, a_max=r_max)
    cos_cutoff = 0.5 * (ops.cos(np.pi * dr_clipped / r_max) + 1.0)
    cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")
    return cutoff


def radial_basis_impl(basis, Z_i, Z_j, embeddings, embed_norm):
    if embeddings is None:
        radial_function = basis
    else:
        # coeffs shape: n_neighbors x n_radialx n_basis
        # reverse convention to match original
        species_pair_coeffs = embeddings[Z_j, Z_i, ...]
        species_pair_coeffs = embed_norm * species_pair_coeffs

        # radial shape: neighbors x n_radial
        radial_function = einops.einsum(
            species_pair_coeffs, basis, "nbrs radial basis, nbrs basis -> nbrs radial"
        )
    return radial_function


class GaussianBasis(nn.Module):
    n_basis: int = 7
    r_min: float = 0.5
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        self.betta = self.n_basis**2 / self.r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
        shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
            self.n_basis
        )

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = jnp.asarray(shifts, dtype=self.dtype)

    def __call__(self, dr):
        basis = gaussian_basis_impl(dr, self.shifts, self.betta, self.rad_norm)
        return basis


class RadialFunction(nn.Module):
    n_radial: int = 5
    basis_fn: nn.Module = GaussianBasis()
    n_species: int = 119
    emb_init: str = "uniform"
    dtype: Any = jnp.float32

    def setup(self):
        self.r_max = self.basis_fn.r_max
        self.embed_norm = jnp.array(
            1.0 / np.sqrt(self.basis_fn.n_basis), dtype=self.dtype
        )
        self.embeddings = None
        if self.emb_init is not None:
            self._n_radial = self.n_radial
            if self.emb_init == "uniform":
                emb_initializer = uniform_range(-1.0, 1.0, dtype=self.dtype)
                self.embeddings = self.param(
                    "atomic_type_embedding",
                    emb_initializer,
                    (
                        self.n_species,
                        self.n_species,
                        self.n_radial,
                        self.basis_fn.n_basis,
                    ),
                    self.dtype,
                )
            else:
                raise NotImplementedError(
                    "Currently only uniformly initialized embeddings and no embeddings"
                    " are implemented."
                )
        else:
            self._n_radial = self.basis_fn.n_basis

    def __call__(self, dr, Z_i, Z_j):
        dr = dr.astype(self.dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        radial_function = radial_basis_impl(basis, Z_i, Z_j, self.embeddings, self.embed_norm)
        cutoff = cosine_cutoff(dr, self.r_max)
        radial_function = radial_function * cutoff

        assert radial_function.dtype == self.dtype
        return radial_function
