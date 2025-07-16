from typing import Callable, Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, TypeAdapter


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

    def create(self, system) -> Callable:
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


class FixLayer(ConstraintBase, extra="forbid"):
    """ """

    name: Literal["fixlayer"] = "fixlayer"
    upper_limit: float
    lower_limit: float

    def create(self, system) -> Callable:
        if jnp.any(system.box != 0):
            cart_pos = system.positions @ system.box

        z_coordinates = cart_pos[:, 2]

        indices = jnp.where(
            (self.lower_limit <= z_coordinates) & (z_coordinates <= self.upper_limit)
        )
        indices = indices[0]

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


Constraint = TypeAdapter(Union[FixAtoms, FixLayer]).validate_python
