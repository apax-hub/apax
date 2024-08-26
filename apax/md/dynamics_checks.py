from typing import Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, TypeAdapter


class DynamicsCheckBase(BaseModel):
    def check(self, predictions):
        pass


class EnergyUncertaintyCheck(DynamicsCheckBase, extra="forbid"):
    name: Literal["energy_uncertainty"] = "energy_uncertainty"
    threshold: float
    per_atom: bool = True

    def check(self, predictions):
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

    def check(self, predictions):
        if "forces_uncertainty" not in predictions.keys():
            m = "No force uncertainties found. Are you using a model ensemble?"
            raise ValueError(m)

        forces_uncertainty = predictions["forces_uncertainty"]

        check_passed = jnp.all(forces_uncertainty < self.threshold)
        return check_passed


DynamicsChecks = TypeAdapter(
    Union[EnergyUncertaintyCheck, ForceUncertaintyCheck]
).validate_python
