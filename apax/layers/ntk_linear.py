from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp


class NTKLinearFlax(nn.Module):
    units: int
    b_init: str = "normal"
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        inputs = inputs.astype(self.dtype)

        w_initializer = nn.initializers.normal(1.0, dtype=self.dtype)

        if self.b_init == "normal":
            b_initializer = nn.initializers.normal(1.0, dtype=self.dtype)
        elif self.b_init == "zeros":
            b_initializer = nn.initializers.constant(0.0, dtype=self.dtype)
        else:
            raise NotImplementedError(
                "Only random normal and zeros intialization of the bias is supported."
            )
        w = self.param("w", w_initializer, (inputs.shape[0], self.units), self.dtype)
        b = self.param("b", b_initializer, [self.units], self.dtype)

        bias_factor = 0.1
        weight_factor = jnp.sqrt(1.0 / inputs.shape[0])

        wx = jnp.dot(inputs, w)
        prediction = weight_factor * wx + bias_factor * b
        assert prediction.dtype == self.dtype
        return prediction
