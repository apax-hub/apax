from typing import Optional

import einops
import haiku as hk
import jax.numpy as jnp
import numpy as np


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
