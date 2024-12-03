from typing import Any

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array
from myrto.so3krates.embedding import (
    ChemicalEmbedding,
    SphericalHarmonics,
)
from myrto.so3krates.so3krates import TransformerBlock
from myrto.utils.safe import masked

from apax.layers.descriptor.basis_functions import BesselBasis
from apax.utils.convert import str_to_dtype


def get_node_mask(Z):
    mask = (Z != 0).astype(jnp.int16)
    return mask


def get_neighbor_mask(idx):
    mask = ((idx[0] - idx[1]) != 0).astype(jnp.int16)
    return mask


class So3kratesRepresentation(nn.Module):
    basis_fn: nn.Module = BesselBasis()
    num_layers: int = 1
    max_degree: int = 3
    num_features: int = 128
    num_heads: int = 4
    use_layer_norm_1: bool = False
    use_layer_norm_2: bool = False
    use_layer_norm_final: bool = False
    activation: str = "silu"
    cutoff_fn: str = "cosine_cutoff"
    transform_input_features: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        dr_vec: Array,
        Z: Array,
        idx: Array,
    ):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)

        R_ij = dr_vec
        i, j = idx[0], idx[1]
        Z_i = Z

        pair_mask = get_neighbor_mask(idx)
        node_mask = get_node_mask(Z)

        scalar_features = ChemicalEmbedding(num_features=self.num_features)(
            Z_i
        )  # -> [nodes, num_features]
        scalar_features *= node_mask[..., None]

        r_ij = e3x.ops.norm(R_ij, axis=-1)  # -> [pairs]

        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)
        cutoffs = cutoff_fn(r_ij, cutoff=self.basis_fn.r_max)  # -> [pairs]
        cutoffs *= pair_mask

        neighborhood_sizes = jax.ops.segment_sum(cutoffs, i, num_segments=Z_i.shape[0])[
            i
        ]  # -> [pairs]
        pair_scale = masked(
            lambda x: 1 / x, neighborhood_sizes[:, None], pair_mask
        ).reshape(-1)

        # radial_expansion = RadialEmbedding(self.num_radial_features, self.cutoff)(r_ij)
        radial_expansion = self.basis_fn(r_ij)
        radial_expansion *= cutoffs[..., None]  # -> [pairs, num_radial_features]

        spherical_expansion = SphericalHarmonics(self.max_degree)(
            R_ij
        )  # -> [pairs, (max_degree+1)**2]
        spherical_expansion *= pair_scale[..., None]

        spherical_features = jax.ops.segment_sum(
            spherical_expansion,
            i,
            num_segments=Z_i.shape[0],
        )  # -> [nodes, (max_degree+1)**2]
        spherical_features *= node_mask[..., None]

        for _ in range(self.num_layers):
            scalar_features, spherical_features = TransformerBlock(
                self.num_heads,
                self.use_layer_norm_1,
                self.use_layer_norm_2,
                activation=self.activation,
                transform_input_features=self.transform_input_features,
            )(
                scalar_features,
                spherical_features,
                radial_expansion,
                spherical_expansion,
                i,
                j,
                pair_mask,
                node_mask,
                cutoffs,
                pair_scale,
            )

        if self.use_layer_norm_final:
            scalar_features = masked(nn.LayerNorm(), scalar_features, node_mask)

        return scalar_features
