from typing import Any, Optional

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap
from jax_md import space

from apax.layers.descriptor.basis_functions import RadialFunction
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype


class GaussianMomentDescriptor(nn.Module):
    radial_fn: nn.Module = RadialFunction()
    n_contr: int = 8
    n_radial_tensor: Optional[int] = None
    dtype: Any = jnp.float32
    apply_mask: bool = True

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(space.distance, 0, 0)

        n_tensor = self.n_radial_tensor or self.n_radial
        if self.n_radial_tensor:
            # one learned map from the radial channels to the (fewer) channels of all l>0 moments,
            # applied per pair; initialised to keep the first n_radial_tensor channels
            self.radial_compression = nn.Dense(
                n_tensor,
                use_bias=False,
                kernel_init=lambda key, shape, dtype: jnp.eye(*shape, dtype=dtype),
                dtype=str_to_dtype(self.dtype),
                param_dtype=str_to_dtype(self.dtype),
                precision=jax.lax.Precision.HIGHEST,
            )

        self.triang_idxs_2d = tril_2d_indices(n_tensor)
        self.triang_idxs_3d = tril_3d_indices(n_tensor)

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
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

        if self.n_radial_tensor:
            moments = geometric_moments(self.radial_compression(radial_function), dn, idx_j, n_atoms)
            moments[0] = jax.ops.segment_sum(radial_function, idx_j, n_atoms)
        else:
            moments = geometric_moments(radial_function, dn, idx_j, n_atoms)

        contr_0 = moments[0]
        contr_1 = jnp.einsum("ari, asi -> ars", moments[1], moments[1])  # noqa: E501
        contr_2 = jnp.einsum("arij, asij -> ars", moments[2], moments[2])  # noqa: E501
        contr_3 = jnp.einsum("arijk, asijk -> ars", moments[3], moments[3])  # noqa: E501
        contr_4 = jnp.einsum(
            "arij, asik, atjk -> arst", moments[2], moments[2], moments[2]
        )  # noqa: E501
        contr_5 = jnp.einsum("ari, asj, atij -> arst", moments[1], moments[1], moments[2])  # noqa: E501
        contr_6 = jnp.einsum(
            "arijk, asijl, atkl -> arst", moments[3], moments[3], moments[2]
        )  # noqa: E501
        contr_7 = jnp.einsum(
            "arijk, asij, atk -> arst", moments[3], moments[2], moments[1]
        )  # noqa: E501

        tril_2_i, tril_2_j = self.triang_idxs_2d[:, 0], self.triang_idxs_2d[:, 1]
        tril_3_i, tril_3_j, tril_3_k = (
            self.triang_idxs_3d[:, 0],
            self.triang_idxs_3d[:, 1],
            self.triang_idxs_3d[:, 2],
        )

        contr_1 = contr_1[:, tril_2_i, tril_2_j]
        contr_2 = contr_2[:, tril_2_i, tril_2_j]
        contr_3 = contr_3[:, tril_2_i, tril_2_j]
        contr_4 = contr_4[:, tril_3_i, tril_3_j, tril_3_k]
        contr_5 = contr_5[:, tril_2_i, tril_2_j]
        contr_6 = contr_6[:, tril_2_i, tril_2_j]

        contr_5 = jnp.reshape(contr_5, [n_atoms, -1])
        contr_6 = jnp.reshape(contr_6, [n_atoms, -1])
        contr_7 = jnp.reshape(contr_7, [n_atoms, -1])

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
        gaussian_moments = jnp.concatenate(gaussian_moments[: self.n_contr], axis=-1)
        assert gaussian_moments.dtype == dtype
        return gaussian_moments
