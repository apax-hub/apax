import jax
import numpy as np
import pytest
from ase import Atoms

from apax.md.bias import SphericalWall, apply_bias_energy

test_wall_bias_data = [
    (
        Atoms("H2", [[0, 0, 0], [0, 0, 10]]),
        5.0,
        1.0,
        (12.5, np.array([[0, 0, 0], [0, 0, 5.0]])),
    ),
    (
        Atoms("H2", [[-10, 0, 0], [0, 0, 10]]),
        5.0,
        1.0,
        (25, np.array([[-5.0, 0, 0], [0, 0, 5.0]])),
    ),
    (
        Atoms("H2", [[-10, 0, 0], [0, 0, 9]]),
        8.0,
        2.0,
        (5, np.array([[-4.0, 0, 0], [0, 0, 2.0]])),
    ),
]


@pytest.mark.parametrize(
    "atoms, radius, spring_constant, energy_and_force", test_wall_bias_data
)
def test_spherical_wall(
    atoms, radius, spring_constant, energy_and_force
) -> tuple[float, np.ndarray]:
    def null_model(R, neighbor, box) -> float:
        return 0.0

    wall_bias = SphericalWall(radius=radius, spring_constant=spring_constant)

    energy_fn = apply_bias_energy(wall_bias, null_model)

    ef_function = jax.value_and_grad(energy_fn)
    energy, force = ef_function(atoms.positions, atoms.get_atomic_numbers(), None)

    assert energy == energy_and_force[0]
    assert np.all(force == energy_and_force[1])
