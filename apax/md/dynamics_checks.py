import logging
from typing import Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, TypeAdapter

log = logging.getLogger(__name__)


class DynamicsCheckBase(BaseModel):
    def check(self, predictions, positions, box):
        pass


class EnergyUncertaintyCheck(DynamicsCheckBase, extra="forbid"):
    name: Literal["energy_uncertainty"] = "energy_uncertainty"
    threshold: float
    per_atom: bool = True

    def check(self, predictions, positions, box):
        if "energy_uncertainty" not in predictions.keys():
            m = "No energy uncertainty found. Are you using a model ensemble?"
            raise ValueError(m)

        energy_uncertainty = predictions["energy_uncertainty"]
        if self.per_atom:
            n_atoms = predictions["forces"].shape[0]
            energy_uncertainty = energy_uncertainty / n_atoms

        check_passed = jnp.all(energy_uncertainty < self.threshold)
        return check_passed


class ForceUncertaintyCheck(DynamicsCheckBase, extra="forbid"):
    name: Literal["forces_uncertainty"] = "forces_uncertainty"
    threshold: float

    def check(self, predictions, positions, box):
        if "forces_uncertainty" not in predictions.keys():
            m = "No force uncertainties found. Are you using a model ensemble?"
            raise ValueError(m)

        forces_uncertainty = predictions["forces_uncertainty"]

        check_passed = jnp.all(forces_uncertainty < self.threshold)
        return check_passed


class ReflectionCheck(DynamicsCheckBase, extra="forbid"):
    name: Literal["reflection"] = "reflection"
    cutoff_plane_height: float

    def check(self, predictions, positions, box):
        cartesian = positions @ box
        z_pos = cartesian[:, 2]

        check_passed = jnp.all(z_pos < self.cutoff_plane_height)

        return check_passed


DynamicsChecks = TypeAdapter(
    Union[EnergyUncertaintyCheck, ForceUncertaintyCheck, ReflectionCheck]
).validate_python
