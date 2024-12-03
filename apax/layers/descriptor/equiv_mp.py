import functools
from typing import Any

import e3x
import flax.linen as nn
import jax
import jax.numpy as jnp

from apax.layers.descriptor.basis_functions import BesselBasis
from apax.layers.masking import mask_by_neighbor
from apax.utils.convert import str_to_dtype


class EquivMPRepresentation(nn.Module):
    basis_fn: nn.Module = BesselBasis()
    features: int = 32
    max_degree: int = 2
    num_iterations: int = 3
    max_atomic_number: int = 118
    dtype: Any = jnp.float32
    apply_mask: bool = True

    @nn.compact
    def __call__(self, dr_vec, Z, neighbor_idxs):
        dtype = str_to_dtype(self.dtype)
        dr_vec = dr_vec.astype(dtype)

        idx_i, idx_j = neighbor_idxs[0], neighbor_idxs[1]
        # 1. Calculate displacement vectors.
        displacements = dr_vec  # Shape (num_pairs, 3).

        # 2. Expand displacement vectors in basis functions.
        basis = (
            e3x.nn.basis(  # Shape (num_pairs, 1, (max_degree+1)**2, num_basis_functions).
                displacements,
                num=self.basis_fn.n_basis,
                max_degree=self.max_degree,
                radial_fn=lambda x, y: self.basis_fn(x),
                cutoff_fn=functools.partial(
                    e3x.nn.smooth_cutoff, cutoff=self.basis_fn.r_max
                ),
            )
        )

        if self.apply_mask:
            basis = mask_by_neighbor(basis, neighbor_idxs)

        # 3. Embed atomic numbers in feature space, x has shape (num_atoms, 1, 1, features).
        x = e3x.nn.Embed(
            num_embeddings=self.max_atomic_number + 1, features=self.features
        )(Z)
        x = x.astype(dtype)

        # 4. Perform iterations (message-passing + atom-wise refinement).
        for i in range(self.num_iterations):
            # Message-pass.
            if i == self.num_iterations - 1:  # Final iteration.
                # Since we will only use scalar features after the final message-pass, we do not want to produce non-scalar
                # features for efficiency reasons.
                y = e3x.nn.MessagePass(max_degree=0, include_pseudotensors=False)(
                    x, basis, dst_idx=idx_i, src_idx=idx_j
                )
                # After the final message pass, we can safely throw away all non-scalar features.
                x = e3x.nn.change_max_degree_or_type(
                    x, max_degree=0, include_pseudotensors=False
                )
            else:
                # In intermediate iterations, the message-pass should consider all possible coupling paths.
                y = e3x.nn.MessagePass()(x, basis, dst_idx=idx_i, src_idx=idx_j)

            x = x.astype(dtype)
            y = y.astype(dtype)
            y = e3x.nn.add(x, y)

            # Atom-wise refinement MLP.
            y = e3x.nn.Dense(self.features)(y)
            y = e3x.nn.silu(y)
            y = e3x.nn.Dense(self.features, kernel_init=jax.nn.initializers.zeros)(y)

            # Residual connection.
            x = e3x.nn.add(x, y)

        x = x[:, 0, 0, :]
        return x
