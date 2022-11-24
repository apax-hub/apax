import jax.numpy as jnp
from jax import grad, random
from jax.config import config

config.update("jax_enable_x64", True)

from jax import ops
from jax_md import partition, space

f32 = jnp.float32
f64 = jnp.float64


def harmonic_morse(dr, D0=5.0, alpha=5.0, r0=1.0, k=50.0, **kwargs):
    dr = jnp.linalg.norm(dr, ord=2, axis=-1)
    U = jnp.where(
        dr < r0,
        0.5 * k * (dr - r0) ** 2 - D0,
        D0 * (jnp.exp(-2.0 * alpha * (dr - r0)) - 2.0 * jnp.exp(-alpha * (dr - r0))),
    )
    return jnp.array(U, dtype=dr.dtype)


N = 4
dimension = 2
box_size = 1  # 6.8
r_cutoff = 1.5

key = random.PRNGKey(0)
key, split = random.split(key)
R = random.uniform(split, (N, dimension), minval=0.0, maxval=box_size, dtype=f64)


# To compare the results, either comment-in the next line or use the mask with M below
# R = R[:2]
print(R)
print(R.shape)


nl_format = partition.Sparse
displacement_fn, shift_fn = space.periodic(box_size)
neighbor_fn = partition.neighbor_list(
    displacement_fn, box_size, r_cutoff, format=nl_format
)

neighbors = neighbor_fn.allocate(R, extra_capacity=0)


M = 2
print(neighbors.idx)

mask = partition.neighbor_list_mask(neighbors)
print(mask)

mask = jnp.logical_and(neighbors.idx[0] < M, neighbors.idx[1] < M)  # & mask
# mask = (neighbors.idx[0] < N) & mask[neighbors.idx[1]]
print(mask)

intmask = mask.astype(int)
n_neighbors = jnp.sum(intmask)
print(n_neighbors)

bond_fn = space.map_bond(displacement_fn)


def energy_fn(R, neighbors, mask):
    R_i, R_j = R[neighbors.idx[0]], R[neighbors.idx[1]]
    dR = bond_fn(R_i, R_j)
    E_ij = harmonic_morse(dR)
    E_ij_masked = E_ij * mask
    E_i = ops.segment_sum(E_ij_masked, neighbors.idx[0], N)
    E_tot = jnp.sum(E_i)
    return E_tot


E = energy_fn(R, neighbors, mask)

print(E)
grad_fn = grad(energy_fn)

F = -grad_fn(R, neighbors, mask)
print(F)
