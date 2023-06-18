import jax
import jax.numpy as jnp
from jax_md.util import Array


def stress_times_vol(energy_fn, position: Array, box, **kwargs) -> Array:
    """Computes the internal stress of a system multiplied with the box volume.
    For training purposes.

    Args:
    energy_fn: A function that computes the energy of the system. This
        function must take as an argument `perturbation` which perturbs the
        box shape. Any energy function constructed using `smap` or in `energy.py`
        with a standard space will satisfy this property.
    position: An array of particle positions.
    box: A box specifying the shape of the simulation volume. Used to infer the
        volume of the unit cell.

    Returns:
    A float specifying the pressure of the system.
    """
    dim = position.shape[1]
    zero = jnp.zeros((dim, dim), position.dtype)
    identity = jnp.eye(dim, dtype=position.dtype)

    def U(eps):
        return energy_fn(position, box=box, perturbation=(identity + eps), **kwargs)

    dUdV = jax.grad(U)
    return -dUdV(zero)