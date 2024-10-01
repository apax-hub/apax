from typing import Literal, Union

import jax.numpy as jnp
from pydantic import BaseModel, TypeAdapter



class ConstraintBase(BaseModel):
    def create(self, state):
        pass



class FixAtoms(ConstraintBase, extra="forbid"):
    name: Literal["fixatoms"] = "fixatoms"
    indices: list[int]

    def create(self, ref_state):

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


        



Constraint = TypeAdapter(
    Union[FixAtoms]
).validate_python
