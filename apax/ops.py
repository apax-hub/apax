import jax
import torch


def swish(x):
    if isinstance(x, jax.Array):
        return jax.nn.swish(x)
    elif isinstance(x, torch.Tensor):
        return torch.nn.functional.silu(x)


def exp(x):
    if isinstance(x, jax.Array):
        return jax.numpy.exp(x)
    elif isinstance(x, torch.Tensor):
        return torch.exp(x)


def log(x):
    if isinstance(x, jax.Array):
        return jax.numpy.log(x)
    elif isinstance(x, torch.Tensor):
        return torch.log(x)


def clip(x, a_min=None, a_max=None):
    if isinstance(x, jax.Array):
        return jax.numpy.clip(x, a_min=a_min, a_max=a_max)
    elif isinstance(x, torch.Tensor):
        return torch.clamp(x, min=a_min, max=a_max)


def cos(x):
    if isinstance(x, jax.Array):
        return jax.numpy.cos(x)
    elif isinstance(x, torch.Tensor):
        return torch.cos(x)


def softplus(x):
    if isinstance(x, jax.Array):
        return jax.nn.softplus(x)
    elif isinstance(x, torch.Tensor):
        return torch.nn.functional.softplus(x)


def sqrt(x):
    if isinstance(x, jax.Array):
        return jax.numpy.sqrt(x)
    elif isinstance(x, torch.Tensor):
        return torch.sqrt(x)


def sum(x, axis, keepdims=False):
    if isinstance(x, jax.Array):
        return jax.numpy.sum(x, axis=axis, keepdims=keepdims)
    elif isinstance(x, torch.Tensor):
        return torch.sum(x, dim=axis, keepdim=keepdims)


def concatenate(x, axis):
    if isinstance(x, jax.Array):
        return jax.numpy.concatenate(x, axis=axis)
    elif isinstance(x, torch.Tensor):
        return torch.concatenate(x, dim=axis)


def dot(x, y):
    if isinstance(x, jax.Array):
        return jax.numpy.dot(x, y)
    elif isinstance(x, torch.Tensor):
        return torch.dot(x, y)


def cast(x, dtype):
    if isinstance(x, jax.Array):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.type(dtype)


def einsum(pattern, *operands, **kwargs):
    if isinstance(operands[0], jax.Array):
        return jax.numpy.einsum(pattern, *operands, **kwargs)
    elif isinstance(operands[0], torch.Tensor):
        return torch.einsum(pattern, *operands, **kwargs)


def segment_sum(x, segment_ids, num_segments=None):
    if isinstance(x, jax.Array):
        return jax.ops.segment_sum(x, segment_ids, num_segments)
    elif isinstance(x, torch.Tensor):
        # TODO pytorch scatter
        return None
