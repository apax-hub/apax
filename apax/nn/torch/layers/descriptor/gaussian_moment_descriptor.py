from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from apax.nn.impl.gaussian_moment_descriptor import gaussian_moment_impl
from apax.nn.torch.layers.descriptor.basis import RadialFunction
from apax.nn.impl.moments import geometric_moments
from apax.nn.impl.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.utils.jax_md_reduced import space


def distance(dR):
    return torch.sqrt(torch.sum(dR**2, axis=-1))


class GaussianMomentDescriptor(nn.Module):
    def __init__(
        self,
        radial_fn: nn.Module = RadialFunction(),
        n_contr: int = 8,
        dtype: Any = torch.float32,
    ):
        super().__init__()
        self.radial_fn = radial_fn
        self.n_contr = n_contr
        self.dtype = dtype

        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = torch.vmap(distance, 0, 0)

        self.triang_idxs_2d = torch.tensor(tril_2d_indices(self.n_radial))
        self.triang_idxs_3d = torch.tensor(tril_3d_indices(self.n_radial))

    def forward(
        self, dr_vec: torch.Tensor, Z: torch.Tensor, neighbor_idxs: torch.Tensor
    ) -> torch.Tensor:
        dr_vec = dr_vec.type(self.dtype)
        # Z shape n_atoms
        n_atoms = Z.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        dr = self.distance(dr_vec)

        # TODO: maybe try jnp where
        dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        radial_function = self.radial_fn(dr, Z_i, Z_j)

        moments = geometric_moments(radial_function, dn, idx_j, n_atoms)
        gaussian_moments = gaussian_moment_impl(
            moments, self.triang_idxs_2d, self.triang_idxs_3d, self.n_contr
        )

        # # gaussian_moments shape: n_atoms x n_features
        # gaussian_moments = jnp.concatenate(gaussian_moments[: self.n_contr], axis=-1)
        assert gaussian_moments.dtype == self.dtype
        return gaussian_moments
