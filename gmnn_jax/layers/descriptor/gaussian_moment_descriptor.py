from typing import Optional

import einops
import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax_md import space

from gmnn_jax.layers.descriptor.basis_functions import RadialFunction
from gmnn_jax.layers.descriptor.moments import MomentContraction, geometric_moments


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
        use_all_features=True,
        name: Optional[str] = None,
    ):
        super().__init__(name)

        self.n_atoms = n_atoms
        self.r_max = r_max
        self.radial_fn = RadialFunction(
            n_species, n_basis, n_radial, r_min, r_max, emb_init=None, name="radial_fn"
        )
        self.displacement = space.map_bond(displacement)
        self.metric = space.map_bond(
            space.canonicalize_displacement_or_metric(displacement)
        )

        moment_indices = [
            [0],
            [1, 1],
            [2, 2],
            [3, 3],
            [2, 2, 2],
            [1, 1, 2],
            [3, 3, 2],
            [3, 2, 1],
        ]
        # convention: a corresponds to
        # n_atoms, m to n_models, r/s/t to n_radial, i/j/k/l to 3
        contr_indices = [
            "ar -> ra",  # Not used in an einsum call, just listed for clarity
            "ari, asi -> rsa",
            "arij, asij -> rsa",
            "arijk, asijk -> rsa",
            "arij, asik, atjk -> rsta",
            "ari, asj, atij -> rsta",
            "arijk, asijl, atkl -> rsta",
            "arijk, asij, atk -> rsta",
        ]

        if use_all_features:
            # triang indices for last contraction are ignored
            # since it does not have any symmetries
            triang_dims = (0, 2, 2, 2, 3, 2, 2, 0)
        else:
            triang_dims = (0, 2, 2, 2, 3, 3, 3, 3)

        self.moment_contractions = []
        for i in range(8):
            moment_contr = MomentContraction(
                n_radial,
                triang_dims[i],
                moment_indices[i],
                contr_indices[i],
                contr_num=i,
                use_all_features=use_all_features,
            )
            self.moment_contractions.append(moment_contr)

    def __call__(self, R, Z, neighbor):
        # R shape n_atoms x 3
        # Z shape n_atoms

        # shape: neighbors
        Z_i, Z_j = Z[neighbor.idx[0]], Z[neighbor.idx[1]]

        # dr_vec shape: neighbors x 3
        dr_vec = self.displacement(R[neighbor.idx[0]], R[neighbor.idx[1]])
        # dr shape: neighbors
        dr = self.metric(R[neighbor.idx[0]], R[neighbor.idx[1]])

        dr_repeated = einops.repeat(dr, "neighbors -> neighbors 1")
        # normalized distance vectors, shape neighbors x 3
        dn = dr_vec / dr_repeated

        # shape: neighbors
        dr_clipped = jnp.clip(dr, a_max=self.r_max)
        cos_cutoff = 0.5 * (jnp.cos(np.pi * dr_clipped / self.r_max) + 1.0)

        radial_function = self.radial_fn(dr, Z_i, Z_j, cos_cutoff)
        # print(radial_function.shape)

        moments = geometric_moments(radial_function, dn, neighbor.idx[0], self.n_atoms)

        gaussian_moments = []
        for i in range(8):
            gm = self.moment_contractions[i](moments)
            gaussian_moments.append(gm)

        # gaussian_moments shape: n_atoms x n_features
        gaussian_moments = jnp.concatenate(gaussian_moments, axis=-1)
        return gaussian_moments
