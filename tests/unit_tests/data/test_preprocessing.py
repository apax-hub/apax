import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def make_variable_sized_data():
    h2o = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ch4 = Atoms(
        "CH5",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
    )
    return [h2o, ch4]


def make_results(atoms):
    num_atoms = len(atoms)

    results = {
        "energy": -num_atoms * 10_000,
        "forces": np.full((num_atoms, 3), num_atoms),
    }
    atoms.calc = SinglePointCalculator(atoms, **results)

    return atoms


def test_atom_padding():
    pass


def test_nl_padding():
    pass


if __name__ == "main":
    test_atom_padding()
