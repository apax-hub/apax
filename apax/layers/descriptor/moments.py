import einops
import jax


import itertools

import numpy as np

# unique index tuples of fully symmetric rank-2 / rank-3 tensors, and the full -> unique gather maps
_U2 = list(itertools.combinations_with_replacement(range(3), 2))  # 6
_U3 = list(itertools.combinations_with_replacement(range(3), 3))  # 10
_G2 = np.array([[_U2.index(tuple(sorted((i, j)))) for j in range(3)] for i in range(3)])
_G3 = np.array([[[_U3.index(tuple(sorted((i, j, k)))) for k in range(3)] for j in range(3)] for i in range(3)])


def geometric_moments_packed(radial_function, dn, idx_i, n_atoms=None):
    """Same moments as `geometric_moments`, but each pair stores only the unique components of the symmetric
    rank-2 / rank-3 products (3 + 6 + 10 instead of 3 + 9 + 27 per channel); full tensors are rebuilt per atom
    after the scatter-sum. Identical results up to floating-point summation order."""
    import jax.numpy as jnp

    u2 = jnp.stack([dn[:, a] * dn[:, b] for a, b in _U2], axis=-1)  # (pairs, 6)
    u3 = jnp.stack([dn[:, a] * dn[:, b] * dn[:, c] for a, b, c in _U3], axis=-1)  # (pairs, 10)
    r = radial_function[..., None]
    m0 = jax.ops.segment_sum(radial_function, idx_i, n_atoms)
    m1 = jax.ops.segment_sum(r * dn[:, None, :], idx_i, n_atoms)
    p2 = jax.ops.segment_sum(r * u2[:, None, :], idx_i, n_atoms)  # (atoms, channels, 6)
    p3 = jax.ops.segment_sum(r * u3[:, None, :], idx_i, n_atoms)  # (atoms, channels, 10)
    return [m0, m1, p2[..., _G2], p3[..., _G3]]


def geometric_moments(radial_function, dn, idx_i, n_atoms=None):
    # dn shape: neighbors x 3
    # radial_function shape: n_neighbors x n_radial

    # s = spatial dim = 3
    xyz = einops.repeat(dn, "nbrs s -> nbrs 1 s")
    xyz2 = einops.repeat(dn, "nbrs s -> nbrs 1 1 s")
    xyz3 = einops.repeat(dn, "nbrs s -> nbrs 1 1 1 s")

    # shape: n_neighbors x n_radial x (3)^(moment_number)
    # s_i = spatial = 3
    zero_moment = radial_function
    first_moment = einops.repeat(zero_moment, "n r -> n r 1") * xyz
    second_moment = einops.repeat(first_moment, "n r s1 -> n r s1 1") * xyz2
    third_moment = einops.repeat(second_moment, "n r s1 s2 -> n r s1 s2 1") * xyz3

    # shape: n_atoms x n_radial x (3)^(moment_number)
    zero_moment = jax.ops.segment_sum(zero_moment, idx_i, n_atoms)
    first_moment = jax.ops.segment_sum(first_moment, idx_i, n_atoms)
    second_moment = jax.ops.segment_sum(second_moment, idx_i, n_atoms)
    third_moment = jax.ops.segment_sum(third_moment, idx_i, n_atoms)

    moments = [zero_moment, first_moment, second_moment, third_moment]

    return moments


def detrace(moments):
    """Traceless parts of the (fully symmetric) rank-2 and rank-3 moments.

    tr M2 = M0 (n.n = 1) and the rank-3 trace vector repeats M1, so the contractions of non-detraced moments
    re-encode lower-order information. M2 -> M2 - tr(M2) I / 3; M3 -> M3 - (d_ij t_k + d_ik t_j + d_jk t_i) / 5,
    t_k = sum_i M3_iik.
    """
    import jax.numpy as jnp

    m0, m1, m2, m3 = moments
    eye = jnp.eye(3, dtype=m2.dtype)
    tr2 = jnp.einsum("...ii->...", m2)
    m2 = m2 - tr2[..., None, None] * eye / 3.0
    t = jnp.einsum("...iik->...k", m3)
    m3 = m3 - (
        jnp.einsum("ij,...k->...ijk", eye, t) + jnp.einsum("ik,...j->...ijk", eye, t)
        + jnp.einsum("jk,...i->...ijk", eye, t)
    ) / 5.0
    return [m0, m1, m2, m3]
