import numpy as np
from apax.data.preprocessing import get_shrink_wrapped_cell


def test_get_shrink_wrapped_cell():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])

    cell, origin = get_shrink_wrapped_cell(positions)

    expected_origin = np.array([0.0, 0.0, 0.0])
    expected_cell = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])

    assert np.allclose(origin, expected_origin)
    assert np.allclose(cell, expected_cell)
