from typing import Optional

import haiku as hk
import jax.numpy as jnp
from haiku.initializers import Constant, RandomNormal


class NTKLinear(hk.Module):
    def __init__(self, units, b_init="normal", name: Optional[str] = None):
        super().__init__(name)
        self.units = units

        self.w_init = RandomNormal(stddev=1.0, mean=0.0)

        if b_init == "normal":
            self.b_init = RandomNormal(stddev=1.0, mean=0.0)
        elif b_init == "zeros":
            self.b_init = Constant(constant=0.0)
        else:
            raise NotImplementedError(
                "Only random normal and zeros intialization of the bias is supported."
            )

    def __call__(self, inputs):
        w = hk.get_parameter("w", shape=(inputs.shape[0], self.units), init=self.w_init)
        b = hk.get_parameter("b", shape=[self.units], init=self.b_init)

        bias_factor = 0.1
        weight_factor = jnp.sqrt(1.0 / inputs.shape[0])

        wx = jnp.dot(inputs, w)
        prediction = weight_factor * wx + bias_factor * b
        return prediction
