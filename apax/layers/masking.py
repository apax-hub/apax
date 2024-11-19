def mask_by_atom(arr, Z):
    mask = (Z != 0).astype(arr.dtype)
    axes_to_add = len(arr.shape) - 1
    for _ in range(axes_to_add):
        mask = mask[..., None]
    masked_arr = arr * mask
    return masked_arr


def mask_by_neighbor(arr, idx):
    mask = ((idx[0] - idx[1]) != 0).astype(arr.dtype)
    if len(arr.shape) == 2:
        mask = mask[..., None]
    elif len(arr.shape) == 4:
        mask = mask[:, None, None, None]
    return arr * mask
