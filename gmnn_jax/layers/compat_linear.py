from typing import Optional

import haiku as hk
import jax.numpy as jnp

from gmnn_jax.layers.initializers import NTKBias, NTKWeights


class CompatLinear(hk.Module):
    def __init__(self, units, name: Optional[str] = None):
        super().__init__(name)
        self.units = units
        self.w_init = NTKWeights()
        self.b_init = NTKBias()

    def __call__(self, inputs):

        w = hk.get_parameter("w", shape=(inputs.shape[0], self.units), init=self.w_init)
        b = hk.get_parameter("b", shape=[self.units], init=self.b_init)     

        bias_factor = 0.1
        weight_factor =  jnp.sqrt(1.0 / inputs.shape[0])

        wx = jnp.dot(inputs, w)
        prediction = weight_factor * wx + bias_factor *  b
        return prediction
