from typing import Optional

import einops
import haiku as hk


class PerElementScaleShift(hk.Module):
    def __init__(self, scale, shift, n_species, name: Optional[str] = None):
        super().__init__(name)
        shift = einops.repeat(shift, "atoms -> atoms 1")

        self.scale = hk.get_parameter(
            "scale_per_element",
            shape=(n_species, 1),
            init=hk.initializers.Constant(scale),
        )

        self.shift = hk.get_parameter(
            "shift_per_element",
            shape=(n_species, 1),
            init=hk.initializers.Constant(shift),
        )

    def __call__(self, x, Z):
        # x shape: n_atoms x 1
        # Z shape: n_atoms
        # scale[Z] shape: n_atoms x 1
        out = self.scale[Z] * x + self.shift[Z]
        return out
