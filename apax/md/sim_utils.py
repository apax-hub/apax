import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from apax.utils.jax_md_reduced import space


@dataclasses.dataclass
class System:
    atomic_numbers: jnp.array
    masses: jnp.array
    positions: jnp.array
    box: jnp.array
    momenta: Optional[jnp.array]

    @classmethod
    def from_atoms(cls, atoms):
        atomic_numbers = jnp.asarray(atoms.numbers, dtype=jnp.int32)
        masses = jnp.asarray(atoms.get_masses(), dtype=jnp.float64)
        momenta = atoms.get_momenta()

        box = jnp.asarray(atoms.cell.array, dtype=jnp.float64)
        box = box.T
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        if np.any(box > 1e-6):
            positions = space.transform(jnp.linalg.inv(box), positions)

        system = cls(
            atomic_numbers=atomic_numbers,
            masses=masses,
            positions=positions,
            box=box,
            momenta=momenta,
        )

        return system


@dataclasses.dataclass
class SimulationFunctions:
    energy_fn: Callable
    auxiliary_fn: Callable
    shift_fn: Callable
    neighbor_fn: Callable
