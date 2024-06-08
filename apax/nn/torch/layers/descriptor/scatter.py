"""Torchscript compatible implementation of segment sum
Curtesy of the SchNetPack developers.
https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/scatter.py
"""

import torch


def segment_sum(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    return _segment_add(x, idx_i, dim_size, dim)


@torch.jit.script
def _segment_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y
