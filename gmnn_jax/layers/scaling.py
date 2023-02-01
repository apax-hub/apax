from typing import Optional

import einops
import haiku as hk
import jax.numpy as jnp


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
