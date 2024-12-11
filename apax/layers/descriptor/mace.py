from typing import Any

import flax.linen as nn
import jax.numpy as jnp

import haiku.experimental.flax as hkflax


# load MACE haiku model here somewhere



class MACERepresentation(nn.Module):
    param_path: str
    n_contr: int = 8
    dtype: Any = jnp.float32
    apply_mask: bool = True

    def setup(self):
        # we need the return of @hk.transform, so this  https://github.com/ACEsuit/mace-jax/blob/4b899de2101c6e2085ee972aeac0e46a334fd9a0/mace_jax/tools/gin_model.py#L204C5-L209C22
        mace_hk = ... 
        self.mace = hkflax.Module(mace_hk)

    def __call__(self, dr_vec, Z, neighbor_idxs):


        vectors = dr_vec # [n_edges, 3]
        node_z = Z # [n_nodes]
        receivers, senders = neighbor_idxs[0], neighbor_idxs[1] # [n_edges]
    
        # node_energies should have shape N_atoms x N_features
        node_energies = self.mace(vectors, node_z, senders, receivers)

        return node_energies
