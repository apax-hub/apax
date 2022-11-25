from typing import Optional

import einops
import haiku as hk
import jax.numpy as jnp
import numpy as np


class GaussianBasis(hk.Module):
    def __init__(self, n_basis, r_min, r_max, name: Optional[str] = None):
        super().__init__(name)

        self.betta = n_basis**2 / r_max**2
        self.rad_norm = (2.0 * self.betta / np.pi) ** 0.25
        shifts = r_min + (r_max - r_min) / n_basis * np.arange(n_basis)

        # shape: 1 x n_basis
        self.shifts = einops.repeat(shifts, "n_basis -> 1 n_basis")

    def __call__(self, dr):
        dr = einops.repeat(dr, "neighbors -> neighbors 1")
        # 1 x n_basis, neighbors x 1 -> neighbors x n_basis
        distances = self.shifts - dr

        # shape: neighbors x n_basis
        basis = jnp.exp(-self.betta * (distances**2))
        basis = self.rad_norm * basis

        return basis
