import einops


def mask_by_atom(arr, Z):
    mask = (Z != 0).astype(arr.dtype)
    mask = einops.rearrange(mask, "n_atoms -> n_atoms 1")
    masked_arr = arr * mask
    return masked_arr


def mask_by_neighbor(arr, idx):
    mask = ((idx[0] - idx[1]) != 0).astype(arr.dtype)
    if len(arr.shape) == 2:
        mask = mask[..., None]
    elif len(arr.shape) == 4:
        mask = mask[:, None, None, None]
    return arr * mask
