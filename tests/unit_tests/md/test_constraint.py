import pytest
from ase import Atoms
from apax.md.constraints import FixCenterOfMass
from apax.md.sim_utils import System
from apax.utils.jax_md_reduced.simulate import NVEState
import numpy as np

constraint_test_data = [
    (Atoms("H2", [[0, 0, 1], [0, 0, -1]], momenta=[[0, 0, 1], [0, 0, 1]]), None),
    (Atoms("H2", [[0, 0, 1], [0, 0, -1]], momenta=[[0, 0, 1], [0, 0, 0.5]]), None),
    (Atoms("H2", [[0, 0, 1], [0, 0, -1]], momenta=[[0, 0, 1], [0, 0, 0.5]]), None),
    (Atoms("HHe", [[0, 0, 1], [0, 0, -1]], momenta=[[0, 0, 1], [0, 0, 0.5]]), None),
]


@pytest.mark.parametrize("atoms, expected", constraint_test_data)
def test_fix_center_of_mass(atoms: Atoms, expected):
    system = System.from_atoms(atoms)
    constraint = FixCenterOfMass()
    apply_constraint_fn = constraint.create(system)[0]

    state = NVEState(
        atoms.get_positions(),
        atoms.get_momenta(),
        np.array([[0, 0, 0], [0, 0, 0]]),
        atoms.get_masses()[:, None],
    )

    new_state = apply_constraint_fn(state)
    print(new_state)
