from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from apax.nn.torch.layers.descriptor.basis import RadialFunctionT
from apax.nn.impl.moments import geometric_moments
from apax.nn.impl.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.utils.jax_md_reduced import space


# def distance(dR):
#     return torch.sqrt(torch.sum(dR**2, axis=-1))

def segment_sum(x, segment_ids):
    out = torch_scatter.scatter(x, segment_ids, dim=0, reduce="sum")
    return out

def geometric_moments(radial_function, dn, idx_i, n_atoms):
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial
    xyz = dn[:, None, :]
    xyz2 = xyz[..., None, :]
    xyz3 = xyz2[..., None, :]

    # shape: n_neighbors x n_radial x (3)^(moment_number)
    zero_moment = radial_function
    first_moment = zero_moment[..., None]  * xyz
    second_moment = first_moment[..., None] * xyz2
    third_moment = second_moment[..., None]  * xyz3

    # shape: n_atoms x n_radial x (3)^(moment_number)
    zero_moment = segment_sum(zero_moment, idx_i)
    first_moment = segment_sum(first_moment, idx_i)
    second_moment = segment_sum(second_moment, idx_i)
    third_moment = segment_sum(third_moment, idx_i)

    moments = [zero_moment, first_moment, second_moment, third_moment]

    return moments


class GaussianMomentDescriptorT(nn.Module):
    def __init__(
        self,
        radial_fn: nn.Module = RadialFunctionT(),
        n_contr: int = 8,
        dtype: Any = torch.float32,
    ):
        super().__init__()
        self.radial_fn = radial_fn
        self.n_contr = n_contr
        self.dtype = dtype

        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn.n_radial

        # self.distance = distance

        self.triang_idxs_2d = torch.tensor(tril_2d_indices(self.n_radial))
        self.triang_idxs_3d = torch.tensor(tril_3d_indices(self.n_radial))
        self.dummy_n_atoms = torch.tensor(1)

    def forward(
        self, dr_vec: torch.Tensor, Z: torch.Tensor, neighbor_idxs: torch.Tensor
    ) -> torch.Tensor:
        dr_vec = dr_vec.type(self.dtype)
        # Z shape n_atoms
        # n_atoms = Z.size(0)

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        # dr shape: neighbors
        # dr = self.distance(dr_vec)
        dr = torch.linalg.norm(dr_vec, dim=-1)

        # dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
        dr_repeated = (dr + 1e-5)[:, None]
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        radial_function = self.radial_fn(dr, Z_i, Z_j)

        moments = geometric_moments(radial_function, dn, idx_j, self.dummy_n_atoms)
        # gaussian_moments = gaussian_moment_impl(
        #     moments, self.triang_idxs_2d, self.triang_idxs_3d, self.n_contr
        # )

        contr_0 = moments[0]
        contr_1 = torch.einsum("ari, asi -> rsa", [moments[1], moments[1]])
        contr_2 = torch.einsum("arij, asij -> rsa", [moments[2], moments[2]])
        contr_3 = torch.einsum("arijk, asijk -> rsa", [moments[3], moments[3]])
        contr_4 = torch.einsum("arij, asik, atjk -> rsta", [moments[2], moments[2], moments[2]])
        contr_5 = torch.einsum("ari, asj, atij -> rsta", [moments[1], moments[1], moments[2]])
        contr_6 = torch.einsum("arijk, asijl, atkl -> rsta", [moments[3], moments[3], moments[2]])
        contr_7 = torch.einsum("arijk, asij, atk -> rsta", [moments[3], moments[2], moments[1]])

        # n_symm01_features = triang_idxs_2d.shape[0] * n_radial
        tril_2_i, tril_2_j = self.triang_idxs_2d[:, 0], self.triang_idxs_2d[:, 1]
        tril_3_i, tril_3_j, tril_3_k = (
            self.triang_idxs_3d[:, 0],
            self.triang_idxs_3d[:, 1],
            self.triang_idxs_3d[:, 2],
        )

        contr_1 = contr_1[tril_2_i, tril_2_j]
        contr_2 = contr_2[tril_2_i, tril_2_j]
        contr_3 = contr_3[tril_2_i, tril_2_j]
        contr_4 = contr_4[tril_3_i, tril_3_j, tril_3_k]
        contr_5 = contr_5[tril_2_i, tril_2_j]
        contr_6 = contr_6[tril_2_i, tril_2_j]

        n_atoms = contr_0.size(0)
        contr_1 = torch.reshape(contr_1, (n_atoms, -1))
        contr_2 = torch.reshape(contr_2, (n_atoms, -1))
        contr_3 = torch.reshape(contr_3, (n_atoms, -1))
        contr_4 = torch.reshape(contr_4, (n_atoms, -1))
        contr_5 = torch.reshape(contr_5, (n_atoms, -1))
        contr_6 = torch.reshape(contr_6, (n_atoms, -1))
        contr_7 = torch.reshape(contr_7, (n_atoms, -1))

        gaussian_moments = [
            contr_0,
            contr_1,
            contr_2,
            contr_3,
            contr_4,
            contr_5,
            contr_6,
            contr_7,
        ]

        # gaussian_moments shape: n_atoms x n_features
        gaussian_moments = torch.concatenate(gaussian_moments[:self.n_contr], dim=-1)
        assert gaussian_moments.dtype == self.dtype
        return gaussian_moments
