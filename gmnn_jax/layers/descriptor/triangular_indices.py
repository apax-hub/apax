import jax.numpy as jnp
import numpy as np


def tril_2d_indices(n: int):
    indices = np.zeros((int(n * (n + 1) / 2), 2), dtype=int)
    sparse_idx = 0
    for i in range(0, n):
        for j in range(i, n):
            indices[sparse_idx] = i, j
            sparse_idx += 1

    return jnp.asarray(indices)


def tril_3d_indices(n: int):
    indices = np.zeros((int(n * (n + 1) * (n + 2) / 6), 3), dtype=int)
    sparse_idx = 0
    for i in range(0, n):
        for j in range(i, n):
            for k in range(j, n):
                indices[sparse_idx] = i, j, k
                sparse_idx += 1

    return jnp.asarray(indices)
