import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass
class NeighborSpoof:
    idx: jnp.array
