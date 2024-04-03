from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from apax.nn.impl.basis import cosine_cutoff, gaussian_basis_impl, radial_basis_impl
from apax.nn.jax.layers.initializers import uniform_range


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

        radial_function = radial_basis_impl(
            basis, Z_i, Z_j, self.embeddings, self.embed_norm
        )
        cutoff = cosine_cutoff(dr, self.r_max)
        radial_function = radial_function * cutoff

        assert radial_function.dtype == self.dtype
        return radial_function
