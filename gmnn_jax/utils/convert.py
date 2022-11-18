from typing import List

import numpy as np
from ase import Atoms


def convert_atoms_to_arrays(atoms_list: List[Atoms]):
    positions = []
    numbers = []
    energy = []

    for atoms in atoms_list:
        positions.append(atoms.get_positions())
        numbers.append(atoms.get_atomic_numbers())
        energy.append(atoms.get_total_energy())

    inputs = {
        "positions": np.stack(positions),
        "numbers": np.stack(numbers),
    }
    labels = {
        "energy": np.vstack(energy),
    }
    return inputs, labels
