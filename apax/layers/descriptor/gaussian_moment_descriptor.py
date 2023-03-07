from typing import Any, Callable, Optional

import einops
import flax.linen as nn
import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_md import space

from apax.layers.descriptor.basis_functions import RadialFunction, RadialFunctionFlax
from apax.layers.descriptor.moments import geometric_moments
from apax.layers.descriptor.triangular_indices import tril_2d_indices, tril_3d_indices
from apax.layers.masking import mask_by_neighbor


def get_disp_fn(displacement):
    def disp_fn(ri, rj, box):
        return displacement(ri, rj, box=box)

    return disp_fn


class GaussianMomentDescriptor(hk.Module):
    def __init__(
        self,
        displacement,
        n_basis,
        n_radial,
        n_species,
        n_atoms,
        r_min,
        r_max,
        dtype=jnp.float32,
        init_box=np.array([0.0, 0.0, 0.0]),
        apply_mask=True,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.n_atoms = n_atoms
        self.n_radial = n_radial
        self.r_max = r_max
        self.radial_fn = RadialFunction(
            n_species,
            n_basis,
            n_radial,
            r_min,
            r_max,
            emb_init=None,
            dtype=dtype,
            name="radial_fn",
        )
        # TODO maybe move the radial func into call and get
        # n_species and n_atoms from the first input batch
        self.init_box = init_box
        if np.all(self.init_box < 1e-6):
            # displacement function for gas phase training and predicting
            self.displacement = space.map_bond(displacement)
        else:
            # displacement function for training on periodic systems
            displacement = get_disp_fn(displacement)
            self.displacement = vmap(displacement, (0, 0, None), 0)

        self.distance = vmap(space.distance, 0, 0)

        self.triang_idxs_2d = tril_2d_indices(n_radial)
        self.triang_idxs_3d = tril_3d_indices(n_radial)
        self.apply_mask = apply_mask

        self.dtype = dtype

    def __call__(self, R, Z, neighbor, box):
        # if R.dtype != self.dtype:
        R = R.astype(self.dtype)
        # R shape n_atoms x 3
        # Z shape n_atoms

        # shape: neighbors
        Z_i, Z_j = Z[neighbor.idx[0]], Z[neighbor.idx[1]]

        # dr_vec shape: neighbors x 3
        if np.all(self.init_box < 1e-6):
            # distance vector for gas phase training and predicting
            dr_vec = self.displacement(
                R[neighbor.idx[1]],
                R[neighbor.idx[0]],
            )  # reverse conventnion to match TF
        else:
            # distance vector for training on periodic systems
            dr_vec = self.displacement(
                R[neighbor.idx[1]],
                R[neighbor.idx[0]],
                box,
            )  # reverse conventnion to match TF

        # dr shape: neighbors
        dr = self.distance(dr_vec)

        dr_vec = dr_vec.astype(self.dtype)
        dr = dr.astype(self.dtype)

        dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")

        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        # shape: neighbors
        dr_clipped = jnp.clip(dr, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr_clipped / self.r_max) + 1.0)

        cos_cutoff = cos_cutoff.astype(self.dtype)

        radial_function = self.radial_fn(dr, Z_i, Z_j, cos_cutoff)
        if self.apply_mask:
            radial_function = mask_by_neighbor(radial_function, neighbor.idx)

        moments = geometric_moments(radial_function, dn, neighbor.idx[1], self.n_atoms)

        contr_0 = moments[0]
        contr_1 = jnp.einsum("ari, asi -> rsa", moments[1], moments[1])
        contr_2 = jnp.einsum("arij, asij -> rsa", moments[2], moments[2])
        contr_3 = jnp.einsum("arijk, asijk -> rsa", moments[3], moments[3])
        contr_4 = jnp.einsum(
            "arij, asik, atjk -> rsta", moments[2], moments[2], moments[2]
        )
        contr_5 = jnp.einsum("ari, asj, atij -> rsta", moments[1], moments[1], moments[2])
        contr_6 = jnp.einsum(
            "arijk, asijl, atkl -> rsta", moments[3], moments[3], moments[2]
        )
        contr_7 = jnp.einsum(
            "arijk, asij, atk -> rsta", moments[3], moments[2], moments[1]
        )

        n_symm01_features = self.triang_idxs_2d.shape[0] * self.n_radial

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

        contr_5 = np.reshape(contr_5, [n_symm01_features, -1])
        contr_6 = np.reshape(contr_6, [n_symm01_features, -1])
        contr_7 = np.reshape(contr_7, [self.n_radial**3, -1])

        contr_1 = jnp.transpose(contr_1)
        contr_2 = jnp.transpose(contr_2)
        contr_3 = jnp.transpose(contr_3)
        contr_4 = jnp.transpose(contr_4)
        contr_5 = jnp.transpose(contr_5)
        contr_6 = jnp.transpose(contr_6)
        contr_7 = jnp.transpose(contr_7)

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
        gaussian_moments = jnp.concatenate(gaussian_moments, axis=-1)
        assert gaussian_moments.dtype == self.dtype
        return gaussian_moments


class GaussianMomentDescriptorFlax(nn.Module):
    displacement_fn: Callable = space.free()[0]
    radial_fn: nn.Module = RadialFunctionFlax()
    n_contr: int = 8
    dtype: Any = jnp.float32
    apply_mask: bool = True
    init_box: np.array = np.array([0.0, 0.0, 0.0])

    def setup(self):
        self.r_max = self.radial_fn.r_max
        self.n_radial = self.radial_fn._n_radial

        if not np.all(self.init_box < 1e-6):
            # displacement function used for training on periodic systems
            mappable_displacement_fn = get_disp_fn(self.displacement_fn)
            self.displacement = vmap(mappable_displacement_fn, (0, 0, None), 0)
        else:
            # displacement function for gas phase training and predicting
            self.displacement = space.map_bond(self.displacement_fn)

        self.distance = vmap(space.distance, 0, 0)

        self.triang_idxs_2d = tril_2d_indices(self.n_radial)
        self.triang_idxs_3d = tril_3d_indices(self.n_radial)

    def __call__(self, R, Z, neighbor_idxs, box):
        R = R.astype(jnp.float64)
        # R shape n_atoms x 3
        # Z shape n_atoms
        n_atoms = R.shape[0]

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]

        # shape: neighbors
        Z_i, Z_j = Z[idx_i], Z[idx_j]

        # dr_vec shape: neighbors x 3
        if not np.all(self.init_box < 1e-6):
            # distance vector for training on periodic systems
            # reverse conventnion to match TF
            dr_vec = self.displacement(R[idx_j], R[idx_i], box).astype(self.dtype)
        else:
            # reverse conventnion to match TF
            # distance vector for gas phase training and predicting
            dr_vec = self.displacement(R[idx_j], R[idx_i]).astype(self.dtype)

        # dr shape: neighbors
        dr = self.distance(dr_vec).astype(self.dtype)

        # TODO: maybe try jnp where
        dr_repeated = einops.repeat(dr + 1e-5, "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        radial_function = self.radial_fn(dr, Z_i, Z_j)
        if self.apply_mask:
            radial_function = mask_by_neighbor(radial_function, neighbor_idxs)

        moments = geometric_moments(radial_function, dn, idx_j, n_atoms)

        contr_0 = moments[0]
        contr_1 = jnp.einsum("ari, asi -> rsa", moments[1], moments[1])
        contr_2 = jnp.einsum("arij, asij -> rsa", moments[2], moments[2])
        contr_3 = jnp.einsum("arijk, asijk -> rsa", moments[3], moments[3])
        contr_4 = jnp.einsum(
            "arij, asik, atjk -> rsta", moments[2], moments[2], moments[2]
        )
        contr_5 = jnp.einsum("ari, asj, atij -> rsta", moments[1], moments[1], moments[2])
        contr_6 = jnp.einsum(
            "arijk, asijl, atkl -> rsta", moments[3], moments[3], moments[2]
        )
        contr_7 = jnp.einsum(
            "arijk, asij, atk -> rsta", moments[3], moments[2], moments[1]
        )

        n_symm01_features = self.triang_idxs_2d.shape[0] * self.n_radial

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

        contr_5 = np.reshape(contr_5, [n_symm01_features, -1])
        contr_6 = np.reshape(contr_6, [n_symm01_features, -1])
        contr_7 = np.reshape(contr_7, [self.n_radial**3, -1])

        contr_1 = jnp.transpose(contr_1)
        contr_2 = jnp.transpose(contr_2)
        contr_3 = jnp.transpose(contr_3)
        contr_4 = jnp.transpose(contr_4)
        contr_5 = jnp.transpose(contr_5)
        contr_6 = jnp.transpose(contr_6)
        contr_7 = jnp.transpose(contr_7)

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
        assert gaussian_moments.dtype == self.dtype
        return gaussian_moments
