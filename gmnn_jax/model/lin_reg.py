from typing import Optional

import haiku as hk
import jax.numpy as jnp
from haiku.initializers import RandomNormal


class LinReg(hk.Module):
    def __init__(self, n_species, name: Optional[str] = None):
        super().__init__(name)

        self.n_species = n_species

    def __call__(self, inputs):
        print("recompiling!")

        w_init = RandomNormal(stddev=1.0, mean=0.0)

        w = hk.get_parameter("w", shape=(self.n_species,), init=w_init)
        b = hk.get_parameter("b", shape=[], init=jnp.ones)

        prediction = jnp.dot(inputs["numbers"], w) + b
        return prediction
