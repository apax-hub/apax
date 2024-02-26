import dataclasses
from typing import List

import einops
import jax.numpy as jnp

from apax.utils.math import normed_dotp


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
    return (1.0 - dotp) / divisor


def force_angle_div_force_label(
    label: jnp.array, prediction: jnp.array, divisor: float = 1.0
):
    """
    Consine similarity loss function weighted by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    loss = jnp.where(F_0_norm > 1e-6, (1.0 - dotp) / F_0_norm, jnp.zeros_like(dotp))
    return loss


def force_angle_exponential_weight(
    label: jnp.array, prediction: jnp.array, divisor: float = 1.0
) -> jnp.array:
    """
    Consine similarity loss function exponentially scaled by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    return (1.0 - dotp) * jnp.exp(-F_0_norm) / divisor


def stress_tril(label, prediction, divisor=1.0):
    idxs = jnp.tril_indices(3)
    label_tril = label[:, idxs[0], idxs[1]]
    prediction_tril = prediction[:, idxs[0], idxs[1]]
    return (label_tril - prediction_tril) ** 2 / divisor


loss_functions = {
    "molecules": weighted_squared_error,
    "structures": weighted_squared_error,
    "vibrations": weighted_squared_error,
    "cosine_sim": force_angle_loss,
    "cosine_sim_div_magnitude": force_angle_div_force_label,
    "cosine_sim_exp_magnitude": force_angle_exponential_weight,
    "tril": stress_tril,
}


@dataclasses.dataclass
class Loss:
    """
    Represents a single weighted loss function that is constructed from a `name`
    and a type of comparison metric.
    """

    name: str
    loss_type: str
    weight: float = 1.0

    def __post_init__(self):
        if self.loss_type not in loss_functions.keys():
            raise NotImplementedError(
                f"the loss function '{self.loss_type}' is not known."
            )

        if self.name not in ["energy", "forces", "stress", "hessian"]:
            raise NotImplementedError(f"the quantity '{self.name}' is not known.")
        self.loss_fn = loss_functions[self.loss_type]

    def __call__(self, inputs: dict, prediction: dict, label: dict) -> float:
        # TODO we may want to insert an additional `mask` argument for this method
        divisor = self.determine_divisor(inputs["n_atoms"])
        loss = self.loss_fn(label[self.name], prediction[self.name], divisor=divisor)
        return self.weight * jnp.sum(jnp.mean(loss, axis=0))

    def determine_divisor(self, n_atoms: jnp.array) -> jnp.array:
        divisor_id = self.name + "_" + self.loss_type
        divisor_dict = {
            "energy_structures": n_atoms**2,
            "energy_vibrations": n_atoms,
            "forces_structures": einops.repeat(n_atoms, "batch -> batch 1 1"),
            "forces_cosine_sim": einops.repeat(n_atoms, "batch -> batch 1 1"),
            "cosine_sim_div_magnitude": einops.repeat(n_atoms, "batch -> batch 1 1"),
            "forces_cosine_sim_exp_magnitude": einops.repeat(
                n_atoms, "batch -> batch 1 1"
            ),
            "stress_structures": einops.repeat(n_atoms**2, "batch -> batch 1 1"),
            "stress_tril": einops.repeat(n_atoms**2, "batch -> batch 1 1"),
            "stress_vibrations": einops.repeat(n_atoms, "batch -> batch 1 1"),
        }
        divisor = divisor_dict.get(divisor_id, jnp.array(1.0))

        return divisor


@dataclasses.dataclass
class LossCollection:
    loss_list: List[Loss]

    def __call__(self, inputs: dict, predictions: dict, labels: dict) -> float:
        total_loss = 0.0
        for single_loss_fn in self.loss_list:
            loss = single_loss_fn(inputs, predictions, labels)
            total_loss = total_loss + loss

        return total_loss
