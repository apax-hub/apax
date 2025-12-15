from apax.utils.math import center_of_mass
import pytest
import numpy as np

test_com_data = [
    (np.array([[0, 0, 0], [0, 0, 1]]), np.array([1, 1]), np.array([0, 0, 0.5])),
    (np.array([[0, 0, -0.5], [0, 0, 0.5]]), np.array([1, 1]), np.array([0, 0, 0])),
    (np.array([[0, 0, -0.5], [0, 0, 0.5]]), np.array([1, 2]), np.array([0, 0, 0.5 / 3])),
    (
        np.array([[2, 0, -0.5], [1, 0, 0.5]]),
        np.array([1, 2]),
        np.array([4.0 / 3.0, 0, 0.5 / 3]),
    ),
]


@pytest.mark.parametrize("positions, masses, expected", test_com_data)
def test_center_of_mass(positions, masses, expected):
    com = center_of_mass(positions, masses)
    assert np.allclose(com, expected), (
        f"positions: {positions}, masses: {masses}, expected: {expected}, got: {com}"
    )
