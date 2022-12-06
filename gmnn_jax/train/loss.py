import dataclasses
from typing import List

import einops
import jax.numpy as jnp

from gmnn_jax.utils.math import normed_dotp


def weighted_squared_error(
    label: jnp.array, prediction: jnp.array, divisor: float = 1.0
) -> jnp.array:
    """
    Squared error function that allows weighting of
    individual contributions by the number of atoms in the system.
    """
    return (label - prediction) ** 2 / divisor


def force_angle_loss(
    label: jnp.array, prediction: jnp.array, divisor: float = 1.0
) -> jnp.array:
    """
    Consine similarity loss function. Contributions are summed in `Loss`.
    """
    dotp = normed_dotp(label, prediction)
    return 1.0 - dotp


def force_angle_div_force_label(
    label: jnp.array, prediction: jnp.array, divisor: float = 1.0
):
    """
    Consine similarity loss function weighted by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    return (1.0 - dotp) / F_0_norm


def force_angle_exponential_weight(
    label: jnp.array, prediction: jnp.array, divisor: float = 1.0
) -> jnp.array:
    """
    Consine similarity loss function exponentially scaled by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    return (1.0 - dotp) * jnp.exp(F_0_norm)


@dataclasses.dataclass
class Loss:
    """
    Represents a single weighted loss function that is constructed from a `key`
    and a type of comparisson metric.
    """

    key: str
    loss_type: str
    weight: float = 1.0

    def __post_init__(self):
        if self.loss_type == "cosine_sim":
            self.loss_fn = force_angle_loss
        elif self.loss_type == "cosine_sim_div_magnitude":
            self.loss_fn = force_angle_div_force_label
        elif self.loss_type == "cosine_sim_exp_magnitude":
            self.loss_fn = force_angle_exponential_weight
        else:
            self.loss_fn = weighted_squared_error

    def __call__(self, prediction: dict, label: dict) -> float:
        # TODO add stress multiplication with cell volume as dataset.map
        # TODO we may want to insert an additional `mask` argument for this method
        divisor = self.determine_divisor(label["n_atoms"])

        loss = self.loss_fn(label[self.key], prediction[self.key], divisor=divisor)
        return self.weight * jnp.sum(loss)

    def determine_divisor(self, n_atoms: jnp.array) -> jnp.array:
        if self.key == "energy" and self.loss_type == "structures":
            divisor = n_atoms**2
        elif self.key == "energy" and self.loss_type == "vibrations":
            divisor = n_atoms
        elif self.key == "forces" and self.loss_type == "structures":
            divisor = einops.repeat(n_atoms, "batch atoms -> batch atoms 1")
        else:
            divisor = jnp.array(1.0)

        return divisor


@dataclasses.dataclass
class LossCollection:
    loss_list: List[Loss]

    def __call__(self, predictions: dict, labels: dict) -> float:
        total_loss = 0.0
        for single_loss_fn in self.loss_list:
            loss = single_loss_fn(predictions, labels)
            total_loss = total_loss + loss

        return total_loss
