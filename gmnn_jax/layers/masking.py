import jax.numpy as jnp


def mask_by_atom(arr, Z):
    mask = (Z == 1).astype(arr.dtype)
    return arr * mask


def mask_by_neighbor(arr, idx):
    mask = (jnp.sum(idx, axis=0) > 0).astype(arr.dtype)[..., None]
    return arr * mask
