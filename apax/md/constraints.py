from typing import Callable, Literal, Union

import jax
import jax.numpy as jnp
from pydantic import BaseModel, TypeAdapter

from apax.md.sim_utils import System
from apax.utils.math import center_of_mass


class ConstraintBase(BaseModel):
    """Base class for constraints.
    Constraints work by implementing a create method.
    This method accepts a reference state which to
    compare to subsequent ones and returns a callable which applies
    the constraint during simulations.
    """

    def create(self, system) -> Callable:
        pass


class FixAtoms(ConstraintBase, extra="forbid"):
    """ """

    name: Literal["fixatoms"] = "fixatoms"
    indices: list[int]

    def create(self, system: System) -> Callable:
        indices = jnp.array(self.indices, dtype=jnp.int64)

        ref_position = system.positions[indices]

        def fn(state):
            position = state.position
            position = position.at[indices].set(ref_position)

            force = state.force
            zero_force = jnp.zeros_like(ref_position)
            force = force.at[indices].set(zero_force)

            momenta = state.momentum
            zero_momenta = jnp.zeros_like(ref_position)
            momenta = momenta.at[indices].set(zero_momenta)

            state = state.set(position=position, force=force, momentum=momenta)
            return state

        return fn, indices


class FixCenterOfMass(ConstraintBase, extra="forbid"):
    name: Literal["fixcenterofmass"] = "fixcenterofmass"
    position: Union[Literal["initial", "origin"], list[float]] = "initial"

    def create(self, system: System) -> Callable:
        if isinstance(self.position, str):
            if self.position.lower() == "initial":
                ref_com = center_of_mass(system.positions, system.masses)
            elif self.position.lower() == "origin":
                ref_com = jnp.array([0, 0, 0])
        else:
            ref_com = jnp.array(self.position)

        def fn(state):
            masses = state.mass[:, 0]

            position = state.position
            position += ref_com - center_of_mass(position, masses)

            momenta = state.momentum
            velocity_com = jnp.sum(momenta, axis=0) / jnp.sum(masses)
            momenta -= masses[:, None] * velocity_com

            # Eqs. (3) and (7) in https://doi.org/10.1021/jp9722824
            # Have not explicitly tested this yet.
            force = state.force
            force -= (
                masses[:, None]
                / jnp.sum(masses**2)
                * jnp.sum(masses[:, None] * force, axis=0)
            )

            state = state.set(position=position, force=force, momentum=momenta)
            return state

        # We return 0 as a constrained idx, to make sure that the
        # integrator knows that we have 3 dof less.
        return fn, [0]


class FixRotation(ConstraintBase, extra="forbid"):
    name: Literal["fixrotation"] = "fixrotation"

    def create(self, system: System) -> Callable:
        raise NotImplementedError()


class FixLayer(ConstraintBase, extra="forbid"):
    """ """

    name: Literal["fixlayer"] = "fixlayer"
    upper_limit: float
    lower_limit: float

    def create(self, system) -> Callable:
        if jnp.any(system.box > 10e-4):
            cart_pos = system.positions @ system.box

        z_coordinates = cart_pos[:, 2]

        indices = jnp.where(
            (self.lower_limit <= z_coordinates) & (z_coordinates <= self.upper_limit)
        )[0]

        ref_position = system.positions[indices]

        def fn(state):
            position = state.position
            position = position.at[indices].set(ref_position)

            force = state.force
            zero_force = jnp.zeros_like(ref_position)
            force = force.at[indices].set(zero_force)

            momenta = state.momentum
            zero_momenta = jnp.zeros_like(ref_position)
            momenta = momenta.at[indices].set(zero_momenta)

            state = state.set(position=position, force=force, momentum=momenta)
            return state

        return fn, indices


Constraint = TypeAdapter(
    Union[FixAtoms, FixCenterOfMass, FixRotation, FixLayer]
).validate_python
