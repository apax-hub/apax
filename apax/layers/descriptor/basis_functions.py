from typing import Any, Optional

import einops
import flax.linen as nn
import haiku as hk
import jax.numpy as jnp
import numpy as np

from apax.layers.initializers import uniform_range


class GaussianBasis(hk.Module):
    def __init__(
        self, n_basis, r_min, r_max, dtype=jnp.float32, name: Optional[str] = None
    ):
        super().__init__(name)

        self.betta = n_basis**2 / r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
        shifts = r_min + (r_max - r_min) / n_basis * np.arange(n_basis)

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


class RadialFunction(hk.Module):
    def __init__(
        self,
        n_species,
        n_basis,
        n_radial,
        r_min,
        r_max,
        emb_init=None,
        dtype=jnp.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.basis_fn = GaussianBasis(n_basis, r_min, r_max)

        self.embed_norm = jnp.array(1.0 / np.sqrt(n_basis), dtype=dtype)
        self.embeddings = hk.get_parameter(
            "atomic_type_embedding",
            shape=(n_species, n_species, n_radial, n_basis),
            init=hk.initializers.RandomUniform(-1.0, 1.0),
            dtype=dtype,
        )

        self.dtype = dtype

    def __call__(self, dr, Z_i, Z_j, cutoff):
        dr = dr.astype(self.dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        # coeffs shape: n_radialx n_basis
        species_pair_coeffs = self.embeddings[
            Z_j, Z_i, ...
        ]  # reverse convention to match original
        species_pair_coeffs = self.embed_norm * species_pair_coeffs

        # radial shape: neighbors x n_radial
        radial_function = einops.einsum(
            species_pair_coeffs, basis, "nbrs radial basis, nbrs basis -> nbrs radial"
        )
        cutoff = einops.repeat(cutoff, "neighbors -> neighbors 1")
        radial_function = radial_function * cutoff
        assert radial_function.dtype == self.dtype

        return radial_function


class GaussianBasisFlax(nn.Module):
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
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        basis = jnp.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis

        return basis


class RadialFunctionFlax(nn.Module):
    n_radial: int = 5
    basis_fn: nn.Module = GaussianBasisFlax()
    n_species: int = 119
    emb_init = None  # Currently unused
    dtype: Any = jnp.float32

    def setup(self):
        self.r_max = self.basis_fn.r_max
        self.embed_norm = jnp.array(
            1.0 / np.sqrt(self.basis_fn.n_basis), dtype=self.dtype
        )
        emb_initializer = uniform_range(-1.0, 1.0, dtype=self.dtype)
        self.embeddings = self.param(
            "atomic_type_embedding",
            emb_initializer,
            (self.n_species, self.n_species, self.n_radial, self.basis_fn.n_basis),
            self.dtype,
        )

    def __call__(self, dr, Z_i, Z_j):
        dr = dr.astype(self.dtype)
        # basis shape: neighbors x n_basis
        basis = self.basis_fn(dr)

        # coeffs shape: n_radialx n_basis
        species_pair_coeffs = self.embeddings[
            Z_j, Z_i, ...
        ]  # reverse convention to match original
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

        assert radial_function.dtype == self.dtype

        return radial_function
