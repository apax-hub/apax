from typing import List

import jax.numpy as jnp
import numpy as np
from ase import Atoms


def convert_atoms_to_arrays(atoms_list: List[Atoms]):
    positions = []
    numbers = []
    energy = []
    forces = []

    for atoms in atoms_list:
        positions.append(atoms.get_positions())
        numbers.append(atoms.get_atomic_numbers())
        energy.append(atoms.get_total_energy())
        forces.append(atoms.get_forces())

    inputs = {
        "positions": np.stack(positions),
        "numbers": np.stack(numbers),
    }
    labels = {"energy": np.stack(energy), "forces": np.stack(forces)}
    return inputs, labels


def tf_to_jax_dict(data_dict):
    data_dict = {k: jnp.asarray(v) for k, v in data_dict.items()}
    return data_dict
