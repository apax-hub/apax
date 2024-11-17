from typing import Any, Union

import einops
import flax.linen as nn
import jax.numpy as jnp

from apax.utils.convert import str_to_dtype


class PerElementScaleShift(nn.Module):
    n_species: int = 119
    scale: Union[jnp.array, float] = 1.0
    shift: Union[jnp.array, float] = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        scale = jnp.asarray(self.scale)
        shift = jnp.asarray(self.shift)

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

        dtype = str_to_dtype(self.dtype)
        self.scale_param = self.param(
            "scale_per_element", scale_init, (n_species, 1), dtype
        )
        self.shift_param = self.param(
            "shift_per_element", shift_init, (n_species, 1), dtype
        )

    def __call__(self, x, Z):
        dtype = str_to_dtype(self.dtype)
        # x shape: n_atoms x 1
        # Z shape: n_atoms
        # scale[Z] shape: n_atoms x 1
        x = x.astype(dtype)

        out = self.scale_param[Z] * x + self.shift_param[Z]

        assert out.dtype == dtype
        return out
