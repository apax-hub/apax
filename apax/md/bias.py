from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, TypeAdapter

from apax.utils.jax_md_reduced.space import distance


class BiasEnergyBase(BaseModel):
    def energy(
        self,
        R: Array,
        neighbor: Array,
        box: Array,
        pertubation: Optional[Array] = None,
    ) -> Array:
        raise NotImplementedError()


def apply_bias_energy(bias: BiasEnergyBase, model: Callable) -> Callable[..., Array]:
    # Function signature:
    # Array, Array, Array, pertubation -> float
    def energy_fn(
        R: Array,
        neighbor: Array,
        box: Array,
        pertubation: Optional[Array] = None,
    ) -> Array:
        energy = model(R=R, neighbor=neighbor, box=box)
        E_bias = bias.energy(R, neighbor, box, pertubation=pertubation)

        return energy + E_bias

    return energy_fn


def apply_bias_auxiliary(
    bias: BiasEnergyBase, model: Callable
) -> Callable[..., Dict[str, Any]]:
    def aux_fn(
        R: Array,
        Z: Array,
        neighbor: Array,
        box: Array,
        offsets: Array,
    ) -> Dict[str, Array]:
        E_bias, neg_F_bias = jax.value_and_grad(bias.energy)(R, neighbor, box)

        prediction = model(R=R, Z=Z, neighbor=neighbor, box=box, offsets=offsets)

        if "energy_unbiased" not in prediction:
            prediction["energy_unbiased"] = prediction["energy"]
            prediction["forces_unbiased"] = prediction["forces"]

        for key in prediction:
            if "unbiased" in key or "uncertainty" in key:
                continue
            if "forces" in key:
                if "ensemble" in key:
                    prediction[key] = prediction[key] - neg_F_bias[:, :, None]
                else:
                    prediction[key] = prediction[key] - neg_F_bias
            elif "energy" in key:
                prediction[key] = prediction[key] + E_bias
                # if "ensemble" in key:
                # else:
                #     prediction[key] = prediction[key] + E_bias

        return prediction

    return aux_fn


class SphericalWall(BiasEnergyBase):
    radius: float
    spring_constant: float

    def energy(
        self,
        R: Array,
        neighbor: Array,
        box: Array,
        pertubation: Optional[Array] = None,
    ) -> Array:
        distance_outside_radius = jnp.clip(distance(R) - self.radius, a_min=0.0)
        return 0.5 * self.spring_constant * jnp.sum(distance_outside_radius**2)


BiasEnergies = TypeAdapter(Union[SphericalWall]).validate_python
