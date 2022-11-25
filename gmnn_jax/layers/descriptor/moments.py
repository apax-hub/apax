from typing import Optional

import einops
import haiku as hk
import jax
import jax.numpy as jnp

from gmnn_jax.layers.descriptor.triangular_indices import (
    tril_2d_indices,
    tril_3d_indices,
)


def geometric_moments(radial_function, dn, idx_i, n_atoms=None):
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial

    # s = spatial dim = 3
    xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")
    xyz2 = einops.repeat(dn, "nbrs s -> nbrs 1 1 s")
    xyz3 = einops.repeat(dn, "nbrs s -> nbrs 1 1 1 s")

    # shape: n_neighbors x n_radial x (3)^(moment_number)
    # s_i = spatial = 3
    zero_moment = radial_function
    first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
    second_moment = einops.repeat(first_moment, "n r s1 -> n r s1 1") * xyz2
    third_moment = einops.repeat(second_moment, "n r s1 s2 -> n r s1 s2 1") * xyz3

    # shape: n_atoms x n_radial x (3)^(moment_number)
    zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
    first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)
    second_moment = jax.ops.segment_sum(second_moment, idx_i, n_atoms)
    third_moment = jax.ops.segment_sum(third_moment, idx_i, n_atoms)

    moments = [zero_moment, first_moment, second_moment, third_moment]

    return moments


class MomentContraction(hk.Module):
    def __init__(
        self,
        n_radial,
        triang_dims,
        moment_indices,
        contr_indices,
        contr_num,
        use_all_features,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_radial = n_radial

        self.triang_dims = triang_dims
        if triang_dims > 0:
            triang_funcs = {2: tril_2d_indices, 3: tril_3d_indices}
            self.tril_idxs = triang_funcs[triang_dims](n_radial)

        self.moment_indices = moment_indices
        self.contr_indices = contr_indices
        self.contr_num = contr_num

        self.use_full_contrs_5_6 = use_all_features and contr_num in (5, 6)
        self.use_full_contr_7 = use_all_features and contr_num == 7

    def __call__(self, moments):
        required_moments = [moments[i] for i in self.moment_indices]

        # each gm has shape n_atoms x n_features_i
        if self.contr_num == 0:
            gm = required_moments[0]
        else:
            contr = jnp.einsum(self.contr_indices, *required_moments)

            if self.use_full_contrs_5_6:
                n_symm01_features = self.tril_idxs.shape[0] * self.n_radial
                tril_i, tril_j = self.tril_idxs[:, 0], self.tril_idxs[:, 1]
                gm = contr[tril_i, tril_j]
                gm = jnp.reshape(gm, [-1, n_symm01_features])

            elif self.use_full_contr_7:
                # has no symmetries, use complete tensor
                gm = jnp.reshape(contr, [-1, self.n_radial**3])

            elif self.triang_dims == 2:
                # use lower triangular indices of contraction
                tril_i, tril_j = self.tril_idxs[:, 0], self.tril_idxs[:, 1]
                gm = contr[tril_i, tril_j]
                gm = jnp.transpose(gm)

            elif self.triang_dims == 3:
                # use lower triangular indices of contraction
                tril_i, tril_j, tril_k = (
                    self.tril_idxs[:, 0],
                    self.tril_idxs[:, 1],
                    self.tril_idxs[:, 2],
                )
                gm = contr[tril_i, tril_j, tril_k]
                gm = jnp.transpose(gm)

            else:
                raise ValueError(
                    f"unknown contraction with triang_dims = {self.triang_dims}"
                )

        return gm
