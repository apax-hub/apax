from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from apax.layers.initializers import uniform_range
from apax.utils.convert import str_to_dtype


class GaussianBasis(nn.Module):
    n_basis: int = 7
    r_min: float = 0.5
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)

        self.betta = self.n_basis**2 / self.r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
        shifts = self.r_min + (self.r_max - self.r_min) / self.n_basis * np.arange(
            self.n_basis
        )

        # shape: 1 x n_basis
        shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")
        self.shifts = jnp.asarray(shifts, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        basis = jnp.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis

        return basis


class BesselBasis(nn.Module):
    """Non-orthogonalized basis functions of Kocer
    https://doi.org/10.1063/1.5086167
    """

    n_basis: int = 7
    r_max: float = 6.0
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.n = jnp.arange(self.n_basis, dtype=dtype)

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        a = (-1) ** self.n * (jnp.sqrt(2) * np.pi / (self.r_max ** (3 / 2)))
        b = (self.n + 1) * (self.n + 2) / jnp.sqrt((self.n + 1) ** 2 * (self.n + 2) ** 2)
        s1 = jnp.sinc((self.n + 1) * dr / self.r_max)
        s2 = jnp.sinc((self.n + 2) * dr / self.r_max)
        basis = a * b * (s1 + s2)
        return basis


class RadialFunction(nn.Module):
    n_radial: int = 5
    basis_fn: nn.Module = GaussianBasis()
    n_species: int = 119
    emb_init: str = "uniform"
    use_embed_norm: bool = True
    one_sided_dist: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        dtype = str_to_dtype(self.dtype)
        self.r_max = self.basis_fn.r_max
        self.embed_norm = jnp.array(1.0 / np.sqrt(self.basis_fn.n_basis), dtype=dtype)
        if self.one_sided_dist:
            lower_bound = 0.0
        else:
            lower_bound = -1.0

        if self.emb_init is not None:
            self._n_radial = self.n_radial
            if self.emb_init == "uniform":
                emb_initializer = uniform_range(lower_bound, 1.0, dtype=dtype)
                self.embeddings = self.param(
                    "atomic_type_embedding",
                    emb_initializer,
                    (
                        self.n_species,
                        self.n_species,
                        self.n_radial,
                        self.basis_fn.n_basis,
                    ),
                    dtype,
                )
            else:
                raise ValueError(
                    "Currently only uniformly initialized embeddings and no embeddings"
                    " are implemented."
                )
        else:
            self._n_radial = self.basis_fn.n_basis

    def __call__(self, dr, Z_i, Z_j):
        dtype = str_to_dtype(self.dtype)
        dr = dr.astype(dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        if self.emb_init is None:
            radial_function = basis
        else:
            # coeffs shape: n_neighbors x n_radialx n_basis
            species_pair_coeffs = self.embeddings[
                Z_j, Z_i, ...
            ]  # reverse convention to match original
            if self.use_embed_norm:
                species_pair_coeffs = self.embed_norm * species_pair_coeffs

            # radial shape: neighbors x n_radial
            radial_function = einops.einsum(
                species_pair_coeffs, basis, "nbrs radial basis, nbrs basis -> nbrs radial"
            )

        # shape: neighbors
        dr_clipped = jnp.clip(dr, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr_clipped / self.r_max) + 1.0)
        cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")

        radial_function = radial_function * cutoff

        assert radial_function.dtype == dtype

        return radial_function
