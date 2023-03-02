import jax.numpy as jnp

# import numpy as np

# def tril_2d_indices(n: int):
#     indices = np.zeros((int(n * (n + 1) / 2), 2), dtype=int)
#     sparse_idx = 0
#     for i in range(0, n):
#         for j in range(i, n):
#             indices[sparse_idx] = i, j
#             sparse_idx += 1

#     return jnp.asarray(indices)


# def tril_3d_indices(n: int):
#     indices = np.zeros((int(n * (n + 1) * (n + 2) / 6), 3), dtype=int)
#     sparse_idx = 0
#     for i in range(0, n):
#         for j in range(i, n):
#             for k in range(j, n):
#                 indices[sparse_idx] = i, j, k
#                 sparse_idx += 1

#     return jnp.asarray(indices)


def tril_2d_indices(n_radial):
    tril_idxs = []
    for i in range(n_radial):
        tril_idxs.append([i, i])
        for j in range(i + 1, n_radial):
            tril_idxs.append([i, j])
    tril_idxs = jnp.array(tril_idxs)
    return tril_idxs


def tril_3d_indices(n_radial):
    tril_idxs = []
    for i in range(n_radial):
        tril_idxs.append([i, i, i])
        for j in range(n_radial):
            if j != i:
                tril_idxs.append([i, j, j])
        for j in range(i + 1, n_radial):
            for k in range(j + 1, n_radial):
                tril_idxs.append([i, j, k])
    tril_idxs = jnp.array(tril_idxs)
    return tril_idxs
