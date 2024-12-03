import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from apax.layers.masking import mask_by_atom
from apax.layers.readout import AtomisticReadout
from apax.utils.math import fp64_sum


def stress_times_vol(energy_fn, position: Array, box, **kwargs) -> Array:
    """Computes the internal stress of a system multiplied with the box volume.
    For training purposes.

    Parameters
    ----------
    energy_fn:
        A function that computes the energy of the system. This
        function must take as an argument `perturbation` which perturbs the
        box shape. Any energy function constructed using `smap` or in `energy.py`
        with a standard space will satisfy this property.
    position:
        An array of particle positions.
    box:
        A box specifying the shape of the simulation volume. Used to infer the
        volume of the unit cell.

    Returns
    -------
    Array
        A float specifying the stress of the system.
    """
    dim = position.shape[1]
    zero = jnp.zeros((dim, dim), position.dtype)
    zero = 0.5 * (zero + zero.T)
    identity = jnp.eye(dim, dtype=position.dtype)

    def U(eps):
        return energy_fn(position, box=box, perturbation=(identity + eps), **kwargs)

    dUdV = jax.grad(U)
    return dUdV(zero)


class PropertyHead(nn.Module):
    """
    the readout is currently limited to a single number
    """

    pname: str
    readout: nn.Module = AtomisticReadout()
    aggregation: str = "none"
    mode: str = "l0"
    apply_mask: bool = True

    def setup(self):
        n_species = 119
        scale_init = nn.initializers.constant(1.0)
        self.scale = self.param(
            "scale_per_element", scale_init, (n_species, 1), jnp.float64
        )

        shift_init = nn.initializers.constant(0.0)

        self.shift_param = self.param(
            "shift_per_element", shift_init, (n_species, 1), jnp.float64
        )

    def __call__(self, g, R, dr_vec, Z, idx, box):
        h = jax.vmap(self.readout)(g)

        is_ensemble = False
        if jnp.size(h, axis=1) > 1:
            # ensemble detected
            is_ensemble = True
            n_ens = jnp.size(h, axis=1)
            h = h[..., None]
            h = jnp.transpose(h, (1, 0, 2))

        p_i = h * self.scale[Z] + self.shift_param[Z]

        if self.mode == "l0":
            p_i = p_i
        elif self.mode == "l1":
            Rc = R - jnp.mean(R, axis=0, keepdims=True)
            r_hat = Rc / jnp.linalg.norm(Rc, axis=1)[:, None]
            p_i = p_i * R
        elif self.mode == "symmetric_traceless_l2":
            Rc = R - jnp.mean(R, axis=0, keepdims=True)
            r_hat = Rc / jnp.linalg.norm(Rc, axis=1)[:, None]
            r_rt = jnp.einsum("ni, nj -> nij", r_hat, r_hat)
            I = jnp.eye(3)
            symmetrized = 3 * r_rt - I
            p_i = p_i[..., None] * symmetrized
        else:
            raise KeyError("unknown symmetry option")

        if is_ensemble:
            p_i = jnp.swapaxes(p_i, 0, 1)  # natoms, nens, features...

        if self.apply_mask:
            p_i = mask_by_atom(p_i, Z)

        if self.aggregation == "none":
            result = p_i
        elif self.aggregation == "sum":
            result = fp64_sum(p_i, axis=0)
        elif self.aggregation == "mean":
            natoms = R.shape[0]
            result = fp64_sum(p_i, axis=0) / natoms
        else:
            raise KeyError("unknown aggregation")

        output = {self.pname: result}

        if is_ensemble:
            divisor = 1 / (n_ens - 1)
            if self.aggregation == "none":
                result = jnp.swapaxes(result, 0, 1)

            mean = jnp.mean(result, axis=0)
            uncertainty = divisor * fp64_sum((mean - result) ** 2, axis=0)
            output[self.pname] = mean
            output[self.pname + "_uncertainty"] = uncertainty

        return output
