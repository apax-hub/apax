from typing import Iterable, Optional, Union

import jax.numpy as jnp
from jax import Array


def fp64_sum(
    X: Array, axis: Optional[Union[Iterable[int], int]] = None, keepdims: bool = False
):
    dtyp = jnp.float64
    result = jnp.sum(X, axis=axis, dtype=dtyp, keepdims=keepdims)
    return result


def normed_dotp(F_0, F_pred):
    F_0_norm = jnp.linalg.norm(F_0, ord=2, axis=2, keepdims=True)
    F_p_norm = jnp.linalg.norm(F_pred, ord=2, axis=2, keepdims=True)

    F_0_n = jnp.where(F_0_norm > 1e-6, F_0 / F_0_norm, jnp.zeros_like(F_0))
    F_p_n = jnp.where(F_p_norm > 1e-6, F_pred / F_p_norm, jnp.zeros_like(F_pred))

    dotp = jnp.einsum("bai, bai -> ba", F_0_n, F_p_n)
    return dotp


def center_of_mass(positions: Array, masses: Array) -> Array:
    """Calculate the center of mass from arrays of positions and masses.

    Args:
        positions (Array): array of coordinates with shape N*3
        masses (Array): array of point masses with shape N

    Returns:
        Array: center of mass coordinates with shape 3
    """
    return jnp.sum(masses[:, None] * positions, axis=0) / jnp.sum(masses)


def inv_and_det_3x3(Sigma: Array) -> tuple[Array, Array]:
    a00 = Sigma[..., 0, 0]
    a01 = Sigma[..., 0, 1]
    a02 = Sigma[..., 0, 2]
    a10 = Sigma[..., 1, 0]  # Sym: a01
    a11 = Sigma[..., 1, 1]
    a12 = Sigma[..., 1, 2]
    a20 = Sigma[..., 2, 0]  # Sym: a02
    a21 = Sigma[..., 2, 1]  # Sym: a12
    a22 = Sigma[..., 2, 2]

    # 3. Analytical Determinant
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )

    invDet = 1.0 / det
    inv00 = (a11 * a22 - a12 * a21) * invDet
    inv01 = (a02 * a21 - a01 * a22) * invDet
    inv02 = (a01 * a12 - a02 * a11) * invDet

    inv10 = (a12 * a20 - a10 * a22) * invDet
    inv11 = (a00 * a22 - a02 * a20) * invDet
    inv12 = (a10 * a02 - a00 * a12) * invDet

    inv12 = (a02 * a10 - a00 * a12) * invDet
    inv20 = (a10 * a21 - a11 * a20) * invDet  # Same as inv02 if symmetric
    inv21 = (a20 * a01 - a00 * a21) * invDet  # Same as inv12 if symmetric
    inv22 = (a00 * a11 - a01 * a10) * invDet

    # Reconstruct Inverse Matrix (N, 3, 3)
    # Stack is faster than assignment
    row0 = jnp.stack([inv00, inv01, inv02], axis=-1)
    row1 = jnp.stack([inv10, inv11, inv12], axis=-1)
    row2 = jnp.stack([inv20, inv21, inv22], axis=-1)
    Sigma_inv = jnp.stack([row0, row1, row2], axis=-2)
    return Sigma_inv, det
