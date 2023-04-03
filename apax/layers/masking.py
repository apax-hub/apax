import einops
import jax.numpy as jnp


def mask_by_atom(arr, Z):
    mask = (Z != 0).astype(arr.dtype)
    mask = einops.rearrange(mask, "n_atoms -> n_atoms 1")
    masked_arr = arr * mask
    return masked_arr


def mask_by_neighbor(arr, idx):
    # mask = (jnp.sum(idx, axis=0) > 0).astype(arr.dtype)[..., None]
    mask = ((idx[0] - idx[1]) != 0).astype(arr.dtype)
    if len(arr.shape) == 2:
        mask = mask[..., None]
    return arr * mask
