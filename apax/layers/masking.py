import jax.numpy as jnp


def mask_by_atom(arr, Z):
    mask = (Z != 0).astype(arr.dtype)
    axes_to_add = len(arr.shape) - 1
    for _ in range(axes_to_add):
        mask = mask[..., None]
    masked_arr = arr * mask
    return masked_arr


def mask_hessian(hessian, Z):
    mask = (Z != 0).astype(hessian.dtype)
    # create a 4D mask for (N, 3, N, 3)
    mask_4d = mask[:, None, None, None] * mask[None, None, :, None]
    return hessian * mask_4d


def mask_by_neighbor(arr, idx, dr_vec=None):
    """Zero out padded neighbor-list entries.

    Padding is stored as (0, 0) pairs with zero displacement. A pair with equal
    indices but a nonzero displacement is an atom interacting with its own
    periodic image (cells with a lattice vector shorter than the cutoff) and is
    kept when `dr_vec` is given.
    """
    mask = idx[0] != idx[1]
    if dr_vec is not None:
        mask = mask | (jnp.sum(dr_vec**2, axis=-1) > 1e-12)
    mask = mask.astype(arr.dtype)
    if len(arr.shape) == 2:
        mask = mask[..., None]
    elif len(arr.shape) == 4:
        mask = mask[:, None, None, None]
    return arr * mask
