import dataclasses
from typing import List

import einops
import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np

from apax.utils.math import normed_dotp


def weighted_squared_error(
    label: jnp.array,
    prediction: jnp.array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
) -> jnp.array:
    """
    Squared error function that allows weighting of
    individual contributions by the number of atoms in the system.
    """
    label, prediction = label[name], prediction[name]
    return (label - prediction) ** 2 / divisor


def weighted_huber_loss(
    label: jnp.array,
    prediction: jnp.array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
) -> jnp.array:
    """
    Huber loss function that allows weighting of
    individual contributions by the number of atoms in the system.
    """
    label, prediction = label[name], prediction[name]
    if "delta" not in parameters.keys():
        raise KeyError("Huber loss function requires 'delta' parameter")
    delta = parameters["delta"]
    diff = jnp.abs(label - prediction)
    loss = jnp.where(diff > delta, delta * (diff - 0.5 * delta), 0.5 * diff**2)
    return loss / divisor


def crps_loss(
    label: jax.Array,
    prediction: jax.Array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
) -> jax.Array:
    """Computes the CRPS of a gaussian distribution given
    means, targets and standard deviations (uncertainty estimate)
    """
    label = label[name]
    means = prediction[name]
    sigmas = prediction[name + "_uncertainty"]

    sigmas = jnp.clip(sigmas, a_min=1e-6)

    norm_x = (label - means) / sigmas
    cdf = 0.5 * (1 + jsc.special.erf(norm_x / jnp.sqrt(2)))

    normalization = 1 / (jnp.sqrt(2.0 * np.pi))

    pdf = normalization * jnp.exp(-(norm_x**2) / 2.0)

    crps = sigmas * (norm_x * (2 * cdf - 1) + 2 * pdf - 1 / jnp.sqrt(np.pi))

    return crps / divisor


def nll_loss(
    label: jax.Array,
    prediction: jax.Array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
) -> jax.Array:
    """Computes the gaussian NLL loss given
    means, targets and standard deviations (uncertainty estimate)
    """
    label = label[name]
    means = prediction[name]
    sigmas = prediction[name + "_uncertainty"]

    variances = jnp.pow(sigmas, 2)
    eps = 1e-4
    sigma = jnp.sqrt(variances)
    sigma = jnp.clip(sigma, a_min=1e-6)

    x1 = jnp.log(jnp.maximum(vars, eps))
    x2 = (means - label) ** 2 / (jnp.maximum(variances, eps))
    nll = 0.5 * (x1 + x2)

    return nll


def force_angle_loss(
    label: jnp.array,
    prediction: jnp.array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
) -> jnp.array:
    """
    Consine similarity loss function. Contributions are summed in `Loss`.
    """
    label, prediction = label[name], prediction[name]
    dotp = normed_dotp(label, prediction)
    return (1.0 - dotp) / divisor


def force_angle_div_force_label(
    label: jnp.array,
    prediction: jnp.array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
):
    """
    Consine similarity loss function weighted by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    label, prediction = label[name], prediction[name]
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    loss = jnp.where(F_0_norm > 1e-6, (1.0 - dotp) / F_0_norm, jnp.zeros_like(dotp))
    return loss


def force_angle_exponential_weight(
    label: jnp.array,
    prediction: jnp.array,
    name,
    divisor: float = 1.0,
    parameters: dict = {},
) -> jnp.array:
    """
    Consine similarity loss function exponentially scaled by the norm of the force labels.
    Contributions are summed in `Loss`.
    """
    label, prediction = label[name], prediction[name]
    dotp = normed_dotp(label, prediction)
    F_0_norm = jnp.linalg.norm(label, ord=2, axis=2, keepdims=False)
    return (1.0 - dotp) * jnp.exp(-F_0_norm) / divisor


def stress_tril(label, prediction, name, divisor=1.0, parameters: dict = {}):
    label, prediction = label[name], prediction[name]
    idxs = jnp.tril_indices(3)
    label_tril = label[:, idxs[0], idxs[1]]
    prediction_tril = prediction[:, idxs[0], idxs[1]]
    return (label_tril - prediction_tril) ** 2 / divisor


loss_functions = {
    "mse": weighted_squared_error,
    "huber": weighted_huber_loss,
    "cosine_sim": force_angle_loss,
    "cosine_sim_div_magnitude": force_angle_div_force_label,
    "cosine_sim_exp_magnitude": force_angle_exponential_weight,
    "tril": stress_tril,
    "crps": crps_loss,
    "nll": nll_loss,
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
    atoms_exponent: float = 1.0
    parameters: dict = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        if self.loss_type not in loss_functions.keys():
            raise NotImplementedError(
                f"the loss function '{self.loss_type}' is not known."
            )

        if self.name not in ["energy", "forces", "stress"]:
            raise NotImplementedError(f"the quantity '{self.name}' is not known.")
        self.loss_fn = loss_functions[self.loss_type]

    def __call__(self, inputs: dict, prediction: dict, label: dict) -> float:
        # TODO we may want to insert an additional `mask` argument for this method
        divisor = self.determine_divisor(inputs["n_atoms"])
        batch_losses = self.loss_fn(
            label, prediction, self.name, divisor, self.parameters
        )
        loss = self.weight * jnp.sum(jnp.mean(batch_losses, axis=0))
        return loss

    def determine_divisor(self, n_atoms: jnp.array) -> jnp.array:
        # shape: batch
        divisor = n_atoms**self.atoms_exponent

        if self.name in ["forces", "stress"]:
            divisor = einops.repeat(divisor, "batch -> batch 1 1")

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
