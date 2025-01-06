from typing import Any

import einops
import flax.linen as nn
import jax.numpy as jnp
from jax import vmap

from apax.layers.descriptor.basis_functions import BesselBasis, RadialFunction
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype
from apax.utils.jax_md_reduced import space
from apax.layers.ntk_linear import NTKLinear

def make_skew_symmetric(v):

    A = jnp.array([
        [0.0, v[2], -v[1]],
        [-v[2], 0.0, v[0]],
        [v[1], -v[0], 0.0],
    ])
    return A

def make_symmetric_traceless(v):
    v_vt = v @ v.T
    Id = jnp.eye(3)
    S = v_vt - 1/3 *Id
    return S



class TensorNet(nn.Module):
    basis_fn: nn.Module = BesselBasis()
    dtype: Any = jnp.float32
    apply_mask: bool = True

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        self.distance = vmap(space.distance, 0, 0)

        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)

        self.dense0 = NTKLinear(32, "lecun", use_ntk=False)

    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)
        # Z shape n_atoms
        n_atoms = Z.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i, ...], Z[idx_j, ...]
        Z_ij = Z[idx_i, idx_j]

        # dr shape: neighbors
        dr = self.distance(dr_vec)

        dr_clipped = jnp.clip(dr, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(jnp.pi * dr_clipped / self.r_max) + 1.0)
        cutoff = einops.repeat(cos_cutoff, "neighbors -> neighbors 1")

        # TODO: maybe try jnp where
        dr_repeated = einops.repeat(jnp.clip(dr, a_min=1e-5), "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        

        I = jnp.repeat(jnp.diag(3)[None, ...], n_atoms, axis=0)
        A = vmap(make_skew_symmetric, 0,0)(dn)
        S = vmap(make_symmetric_traceless, 0,0)(dn)


        # EQ 7
        rbf = self.basis_fn(dr)

        fI = self.dense0(rbf)
        fA = self.dense1(rbf)
        fS = self.dense2(rbf)

        X_ij = cutoff * Z_ij * (fI * I + fA * A + fS * S)

        

