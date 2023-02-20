from typing import Optional

import einops
import haiku as hk
import jax.numpy as jnp
import flax.linen as nn

class PerElementScaleShift(hk.Module):
    def __init__(
        self, scale, shift, n_species, dtype=jnp.float32, name: Optional[str] = None
    ):
        super().__init__(name)
        if scale is None:
            scale = 1.0
        if shift is None:
            shift = jnp.zeros(n_species)
        if len(shift.shape) == 1:
            shift = einops.repeat(shift, "species -> species 1")

        self.scale = hk.get_parameter(
            "scale_per_element",
            shape=(n_species, 1),
            init=hk.initializers.Constant(scale),
            dtype=dtype,
        )

        self.shift = hk.get_parameter(
            "shift_per_element",
            shape=(n_species, 1),
            init=hk.initializers.Constant(shift),
            dtype=dtype,
        )

        self.dtype = dtype

    def __call__(self, x, Z):
        # x shape: n_atoms x 1
        # Z shape: n_atoms
        # scale[Z] shape: n_atoms x 1
        x = x.astype(self.dtype)
        out = self.scale[Z] * x + self.shift[Z]

        assert out.dtype == self.dtype
        return out


class PerElementScaleShiftFlax(nn.Module):
    n_species: int = 119
    scale: Optional[jnp.array] = 1.0
    shift: Optional[jnp.array] = 0.0
    dtype=jnp.float32
    
    def setup(self):
        scale = jnp.asarray(self.scale)
        shift = jnp.asarray(self.shift)
        
        if scale.shape != shift.shape:
            raise ValueError("Scale and Shift parameters should have the same shape")

        if len(scale.shape) > 0:
            n_species = scale.shape[0]
        else:
            n_species = self.n_species

        if len(scale.shape) == 1:
            scale = einops.repeat(scale, "species -> species 1")
        if len(shift.shape) == 1:
            shift = einops.repeat(shift, "species -> species 1")
        scale_init = nn.initializers.constant(scale)
        shift_init = nn.initializers.constant(shift)

        self.scale_param = self.param('scale_per_element', scale_init, (n_species, 1), self.dtype)
        self.shift_param = self.param('shift_per_element', shift_init, (n_species, 1), self.dtype)

    def __call__(self, x, Z):
        # x shape: n_atoms x 1
        # Z shape: n_atoms
        # scale[Z] shape: n_atoms x 1
        x = x.astype(self.dtype)
        
        out = self.scale_param[Z] * x + self.shift_param[Z]

        assert out.dtype == self.dtype
        return out
