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

    def create(self, ref_stat, system) -> Callable:
        pass


class FixAtoms(ConstraintBase, extra="forbid"):
    """ """

    name: Literal["fixatoms"] = "fixatoms"
    indices: list[int]

    def create(self, ref_state, system) -> Callable:
        indices = jnp.array(self.indices, dtype=jnp.int64)

        ref_position = ref_state.position[indices]

        def fn(state):
            position = state.position
            force = state.force
            zero_force = jnp.zeros_like(ref_position)
            position = position.at[indices].set(ref_position)
            force = force.at[indices].set(zero_force)
            state = state.set(position=position, force=force)
            return state

        return fn


class FixLayer(ConstraintBase, extra="forbid"):
    """ """

    name: Literal["fixlayer"] = "fixlayer"
    upper_limit: float
    lower_limit: float

    def create(self, ref_state, system) -> Callable:
        pos = ref_state.position @ system.box
        z_coordinates = pos[:, 2]

        indices = jnp.where(
            (self.lower_limit <= z_coordinates) & (z_coordinates <= self.upper_limit)
        )

        ref_position = ref_state.position[indices]

        def fn(state):
            position = state.position
            force = state.force
            zero_force = jnp.zeros_like(ref_position)
            position = position.at[indices].set(ref_position)
            force = force.at[indices].set(zero_force)
            state = state.set(position=position, force=force)
            return state

        return fn


Constraint = TypeAdapter(Union[FixAtoms, FixLayer]).validate_python
