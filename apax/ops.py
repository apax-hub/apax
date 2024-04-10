from typing import List, Union
import jax
import torch
import torch_scatter

Array = Union[torch.Tensor, jax.Array]

def swish(x):
    # if isinstance(x, jax.Array):
    #     return jax.nn.swish(x)
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.silu(x)


def exp(x):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.exp(x)
    if isinstance(x, torch.Tensor):
        return torch.exp(x)


def log(x):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.log(x)
    if isinstance(x, torch.Tensor):
        return torch.log(x)


def clip(x, a_max):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.clip(x, a_min=a_min, a_max=a_max)
    if isinstance(x, torch.Tensor):
        # print(a_max)
        return torch.clamp(x, max=a_max)


def cos(x):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.cos(x)
    if isinstance(x, torch.Tensor):
        return torch.cos(x)


def softplus(x):
    # if isinstance(x, jax.Array):
    #     return jax.nn.softplus(x)
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.softplus(x)


def sqrt(x):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.sqrt(x)
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)


def sum(x, axis, keepdims=False):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.sum(x, axis=axis, keepdims=keepdims)
    if isinstance(x, torch.Tensor):
        return torch.sum(x, dim=axis, keepdim=keepdims)


def concatenate(x: list, axis):
    # if isinstance(x[0], jax.Array):
    #     return jax.numpy.concatenate(x, axis=axis)
    if isinstance(x[0], torch.Tensor):
        return torch.concatenate(x, dim=axis)


def dot(x, y):
    # if isinstance(x, jax.Array):
    #     return jax.numpy.dot(x, y)
    if isinstance(x, torch.Tensor):
        return torch.dot(x, y)


def cast(x, dtype):
    # if isinstance(x, jax.Array):
    #     return x.astype(dtype)
    if isinstance(x, torch.Tensor):
        return x.type(dtype)


def einsum(pattern: str, operands: List[torch.Tensor]):
    # print([type(o) for o in operands])
    # quit()
    # if isinstance(operands[0], jax.Array):
    #     return jax.numpy.einsum(pattern, *operands, **kwargs)
    if isinstance(operands[0], torch.Tensor):
        return torch.einsum(pattern, operands)


def segment_sum(x, segment_ids, num_segments):
    # if isinstance(x, jax.Array):
    #     return jax.ops.segment_sum(x, segment_ids, num_segments)
    if isinstance(x, torch.Tensor):
        out = torch_scatter.scatter(x, segment_ids, dim=0, reduce="sum")
        return out
