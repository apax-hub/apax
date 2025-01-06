from typing import Any

import einops
import flax.linen as nn
from jax import vmap
import jax
import jax.numpy as jnp
from apax.layers.descriptor.basis_functions import GaussianBasis
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype
from apax.utils.jax_md_reduced import space


class FirstInteraction(nn.Module):

    def setup(self):
        ...
    
    def __call__(self, dn, h_s, basis, idx_i):
        n_atoms = h_s.shape[0]
        
        # s = spatial dim = 3
        xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")

        # shape: n_neighbors x n_radial x (3)^(moment_number)
        # s_i = spatial = 3
        # TODO multiple with h0
        zero_moment = h_s * basis
        first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
        # TODO linear projection
        moments = [zero_moment, first_moment]

        # shape: n_atoms x n_radial x (3)^(moment_number)
        # TODO self attention
        zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
        first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)


        h_s0 = moments[0]
        h_s1 = jnp.einsum("ari, asi -> ars", moments[1], moments[1])  # noqa: E501
        h_s1 = jnp.reshape(h_s1, [n_atoms, -1])

        h_p = jnp.einsum("ar, asi -> airs", moments[0], moments[1])  # noqa: E501
        h_p = jnp.reshape(h_p, [n_atoms, 3, -1])

        h_s = [h_s0, h_s1]
        h_s = jnp.concatenate(h_s, axis=-1)

        return h_s, h_p
    

class Interaction(nn.Module):

    def setup(self):
        ...
    
    def __call__(self, dn, h_s, basis, idx_i):
        n_atoms = h_s.shape[0]
        
        # s = spatial dim = 3
        xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")

        # shape: n_neighbors x n_radial x (3)^(moment_number)
        # s_i = spatial = 3
        # TODO multiple with h0
        zero_moment = h_s * basis
        first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
        # TODO linear projection
        moments = [zero_moment, first_moment]

        # shape: n_atoms x n_radial x (3)^(moment_number)
        # TODO self attention
        zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
        first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)


        h_s0 = moments[0]
        h_s1 = jnp.einsum("ari, asi -> ars", moments[1], moments[1])  # noqa: E501
        h_s1 = jnp.reshape(h_s1, [n_atoms, -1])

        h_p = jnp.einsum("ar, asi -> airs", moments[0], moments[1])  # noqa: E501
        h_p = jnp.reshape(h_p, [n_atoms, 3, -1])

        h_s = [h_s0, h_s1]
        h_s = jnp.concatenate(h_s, axis=-1)

        return h_s, h_p



class NewModel(nn.Module):
    dtype: Any = jnp.float32
    basis_fn: nn.Module = GaussianBasis()
    apply_mask: bool = True

    def setup(self):

        self.distance = vmap(space.distance, 0, 0)
        self.embedding = Embed
        self.interact0 = FirstInteraction()
        self.interact1 = Interaction()

        

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        # Z shape n_atoms
        n_atoms = Z.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        # Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]

        h_s = self.embedding(Z)
        h_p = jnp.zeros((n_atoms, 3, h_s.shape[-1]))

        # dr shape: neighbors
        dr = self.distance(dr_vec)
        dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        basis = self.basis_fn(dr)
        if self.apply_mask:
            basis = mask_by_neighbor(basis, neighbor_idxs)

        h_s, h_p = self.interact0(dn, h_s, basis, idx_i)
        h_s, h_p = self.interact1(dn, h_s, h_p, basis, idx_i)

        # h_p_norm = jnp.linalg.norm(h_p, axis=1)

        # h = jnp.concatenate([h_s, h_p_norm], axis=-1)
        h = h_s
        return h
