from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
from jax import vmap
from jax_md import space

from apax.layers.descriptor.basis_functions import RadialFunction
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax import ops


def gaussian_moment_impl(moments, triang_idxs_2d, triang_idxs_3d, n_contr):
    
    contr_0 = moments[0]
    contr_1 = ops.einsum("ari, asi -> rsa", moments[1], moments[1])
    contr_2 = ops.einsum("arij, asij -> rsa", moments[2], moments[2])
    contr_3 = ops.einsum("arijk, asijk -> rsa", moments[3], moments[3])
    contr_4 = ops.einsum("arij, asik, atjk -> rsta", moments[2], moments[2], moments[2])
    contr_5 = ops.einsum("ari, asj, atij -> rsta", moments[1], moments[1], moments[2])
    contr_6 = ops.einsum("arijk, asijl, atkl -> rsta", moments[3], moments[3], moments[2])
    contr_7 = ops.einsum("arijk, asij, atk -> rsta", moments[3], moments[2], moments[1])

    # n_symm01_features = triang_idxs_2d.shape[0] * n_radial

    tril_2_i, tril_2_j = triang_idxs_2d[:, 0], triang_idxs_2d[:, 1]
    tril_3_i, tril_3_j, tril_3_k = (
        triang_idxs_3d[:, 0],
        triang_idxs_3d[:, 1],
        triang_idxs_3d[:, 2],
    )

    contr_1 = contr_1[tril_2_i, tril_2_j]
    contr_2 = contr_2[tril_2_i, tril_2_j]
    contr_3 = contr_3[tril_2_i, tril_2_j]
    contr_4 = contr_4[tril_3_i, tril_3_j, tril_3_k]
    contr_5 = contr_5[tril_2_i, tril_2_j]
    contr_6 = contr_6[tril_2_i, tril_2_j]

    contr_1 = einops.rearrange(contr_1, "features atoms -> atoms features")
    contr_2 = einops.rearrange(contr_2, "features atoms -> atoms features")
    contr_3 = einops.rearrange(contr_3, "features atoms -> atoms features")
    contr_4 = einops.rearrange(contr_4, "features atoms -> atoms features")
    contr_5 = einops.rearrange(contr_5, "f1 f2 atoms -> atoms (f1 f2)")
    contr_6 = einops.rearrange(contr_6, "f1 f2 atoms -> atoms (f1 f2)")
    contr_7 = einops.rearrange(contr_7, "f1 f2 f3 atoms -> atoms (f1 f2 f3)")

    # contr_1 = jnp.transpose(contr_1)
    # contr_2 = jnp.transpose(contr_2)
    # contr_3 = jnp.transpose(contr_3)
    # contr_4 = jnp.transpose(contr_4)
    # contr_5 = jnp.reshape(contr_5, [-1, n_symm01_features])
    # contr_6 = jnp.reshape(contr_6, [-1, n_symm01_features])
    # contr_7 = jnp.reshape(contr_7, [-1, n_radial**3])

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
    gaussian_moments = ops.concatenate(gaussian_moments[: n_contr], axis=-1)
    return gaussian_moments


class GaussianMomentDescriptor(nn.Module):
    radial_fn: nn.Module = RadialFunction()
    n_contr: int = 8
    dtype: Any = jnp.float32
    apply_mask: bool = True

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(space.distance, 0, 0)

        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dr_vec = dr_vec.astype(self.dtype)
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
        if self.apply_mask:
            radial_function = mask_by_neighbor(radial_function, neighbor_idxs)

        moments = geometric_moments(radial_function, dn, idx_j, n_atoms)
        gaussian_moments = gaussian_moment_impl(moments, self.triang_idxs_2d, self.triang_idxs_3d, self.n_contr)

        # contr_0 = moments[0]
        # contr_1 = ops.einsum(moments[1], moments[1], "ari, asi -> rsa")
        # contr_2 = ops.einsum(moments[2], moments[2], "arij, asij -> rsa")
        # contr_3 = ops.einsum(moments[3], moments[3], "arijk, asijk -> rsa")
        # contr_4 = ops.einsum(moments[2], moments[2], moments[2], "arij, asik, atjk -> rsta")
        # contr_5 = ops.einsum(moments[1], moments[1], moments[2], "ari, asj, atij -> rsta")
        # contr_6 = ops.einsum(moments[3], moments[3], moments[2], "arijk, asijl, atkl -> rsta")
        # contr_7 = ops.einsum(moments[3], moments[2], moments[1], "arijk, asij, atk -> rsta")

        # n_symm01_features = self.triang_idxs_2d.shape[0] * self.n_radial

        # tril_2_i, tril_2_j = self.triang_idxs_2d[:, 0], self.triang_idxs_2d[:, 1]
        # tril_3_i, tril_3_j, tril_3_k = (
        #     self.triang_idxs_3d[:, 0],
        #     self.triang_idxs_3d[:, 1],
        #     self.triang_idxs_3d[:, 2],
        # )

        # contr_1 = contr_1[tril_2_i, tril_2_j]
        # contr_2 = contr_2[tril_2_i, tril_2_j]
        # contr_3 = contr_3[tril_2_i, tril_2_j]
        # contr_4 = contr_4[tril_3_i, tril_3_j, tril_3_k]
        # contr_5 = contr_5[tril_2_i, tril_2_j]
        # contr_6 = contr_6[tril_2_i, tril_2_j]

        # contr_5 = jnp.reshape(contr_5, [n_symm01_features, -1])
        # contr_6 = jnp.reshape(contr_6, [n_symm01_features, -1])
        # contr_7 = jnp.reshape(contr_7, [self.n_radial**3, -1])
        # # use einops rearrange
        # contr_1 = jnp.transpose(contr_1)
        # contr_2 = jnp.transpose(contr_2)
        # contr_3 = jnp.transpose(contr_3)
        # contr_4 = jnp.transpose(contr_4)
        # contr_5 = jnp.transpose(contr_5)
        # contr_6 = jnp.transpose(contr_6)
        # contr_7 = jnp.transpose(contr_7)
        # # use einops rearrange

        # gaussian_moments = [
        #     contr_0,
        #     contr_1,
        #     contr_2,
        #     contr_3,
        #     contr_4,
        #     contr_5,
        #     contr_6,
        #     contr_7,
        # ]

        # # gaussian_moments shape: n_atoms x n_features
        # gaussian_moments = jnp.concatenate(gaussian_moments[: self.n_contr], axis=-1)
        assert gaussian_moments.dtype == self.dtype
        return gaussian_moments
